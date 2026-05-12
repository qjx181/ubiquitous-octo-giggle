import json
import logging
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from auth import get_token_from_request, decode_access_token
from memory import get_history, append_message, get_history_summary, get_session_role
from llm_client import stream_llm
from services.prompt import build_general_system_prompt, build_professional_rag_prompt
from services.retrieval import build_context, get_query_collection_name, get_query_jsonl_path, should_use_local_knowledge, is_web_query, is_greeting_or_smalltalk
from services.safety import verify_and_enforce_sources
from services.knowledge_store import save_chat_knowledge_to_milvus
from web_fallback import get_last_web_debug, crawl_web_knowledge, build_web_context, save_web_knowledge_to_milvus, web_collection_name
from config import MIN_RETRIEVAL_SCORE

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(req: dict, request: Request):
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    username = decode_access_token(token)
    session_id = (req.get("session_id") or "").strip()
    query = (req.get("query") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")
    if not query:
        raise HTTPException(status_code=400, detail="query不能为空")

    session_role = get_session_role(username, session_id)
    role_to_use = (req.get("role") or session_role).strip()
    style_hint = (req.get("style") or "默认").strip() or "默认"
    history_messages = get_history(username, session_id)
    history_summary = get_history_summary(username, session_id)
    messages = list(history_messages)
    if history_summary:
        messages.insert(0, {"role": "system", "content": f"这是当前会话的历史摘要：\n{history_summary}"})
    messages.append({"role": "user", "content": query})

    rag_context, refs, retrieval_seconds, bm25_item_count, bm25_hit_count, vector_hit_count, best_score = build_context(query, role_to_use)
    general_chat = is_greeting_or_smalltalk(query)
    web_saved_count = 0
    web_context = ""
    web_collection = ""
    web_search_attempted = False
    needs_web_fallback = (not general_chat) and (not bool(rag_context))
    retrieval_fast_enough = retrieval_seconds <= 1.8
    low_score_fallback = (not general_chat) and bool(rag_context) and best_score < MIN_RETRIEVAL_SCORE

    if (needs_web_fallback and not retrieval_fast_enough) or low_score_fallback:
        web_search_attempted = True
        web_collection = web_collection_name(role_to_use)
        web_documents = crawl_web_knowledge(query, role_to_use)
        if web_documents:
            web_context, web_refs = build_web_context(web_documents)
            refs = web_refs
            rag_context = web_context
            best_score = 1.0
            from services.retrieval import get_embedding_model
            web_saved_count = save_web_knowledge_to_milvus(web_documents, get_embedding_model(), web_collection)
        else:
            web_search_attempted = False

    role_hint = "你是一个专业、友好、可靠的中文智能助手。"
    if web_context:
        messages.insert(0, {"role": "system", "content": build_professional_rag_prompt(role_to_use, style_hint, web_context, role_hint, from_web=True)})
    elif rag_context:
        messages.insert(0, {"role": "system", "content": build_professional_rag_prompt(role_to_use, style_hint, rag_context, role_hint, from_web=any(ref.get("source") == "web_memory" for ref in refs))})
    elif general_chat:
        messages.insert(0, {"role": "system", "content": build_general_system_prompt(role_to_use, style_hint, role_hint)})

    def event_stream():
        full_answer = []
        try:
            append_message(username, session_id, "user", query)
            meta_json = json.dumps({"user_id": username, "session_id": session_id, "role": role_to_use, "style": style_hint, "references": refs, "debug_info": {"role": role_to_use, "style": style_hint, "collection": get_query_collection_name(role_to_use), "jsonl_path": get_query_jsonl_path(role_to_use), "references_count": len(refs), "has_rag_context": bool(rag_context), "retrieval_seconds": round(retrieval_seconds, 3), "rag_status": "联网兜底" if web_context else ("普通交流" if general_chat and not rag_context else ("已命中RAG" if rag_context else "未命中RAG")), "general_chat": general_chat, "best_score": round(best_score, 4), "min_retrieval_score": MIN_RETRIEVAL_SCORE, "needs_web_fallback": needs_web_fallback, "use_local_knowledge": should_use_local_knowledge(query, role_to_use), "is_web_query": is_web_query(query, role_to_use), "web_fallback": bool(web_context), "web_collection": web_collection, "web_saved_count": web_saved_count, "web_search_attempted": web_search_attempted, "web_debug": get_last_web_debug()}}, ensure_ascii=False)
            yield f"event: meta\ndata: {meta_json}\n\n"
            for chunk in stream_llm(messages):
                full_answer.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
            raw_answer = "".join(full_answer).strip() or "不足以判断"
            final_answer = verify_and_enforce_sources(raw_answer, rag_context, web_context, refs)
            final_answer = re.sub(r"(?is)^\s*(?:【任务】|\[任务\]|输出格式|答案：|检索来源：).*$", "", final_answer).strip() or final_answer
            append_message(username, session_id, "assistant", final_answer)
            try:
                from services.retrieval import get_embedding_model
                save_chat_knowledge_to_milvus(
                    get_embedding_model(),
                    user_id=username,
                    session_id=session_id,
                    role=role_to_use,
                    style=style_hint,
                    query=query,
                    retrieved_context=web_context or rag_context,
                    answer=final_answer,
                    references=refs,
                )
            except Exception:
                logger.exception("save chat knowledge to milvus failed user_id=%s session_id=%s", username, session_id)
            yield f"event: done\ndata: {json.dumps({'answer': final_answer}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("chat stream failed user_id=%s session_id=%s", username, session_id)
            yield f"event: error\ndata: {json.dumps({'detail': '服务异常'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
