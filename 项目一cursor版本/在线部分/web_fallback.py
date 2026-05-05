import asyncio
import hashlib
import html
import re
import time
import traceback
from datetime import datetime
from typing import Iterable, Optional
from urllib.parse import quote_plus, urljoin
import concurrent.futures
import os
from types import SimpleNamespace

# Optional httpx client for connection pooling and async support
try:
    import httpx
except Exception:
    httpx = None

# Enable httpx client when env USE_HTTPX is truthy and httpx is available
USE_HTTPX = str(os.getenv("USE_HTTPX", "")).lower() in ("1", "true", "yes") and httpx is not None
HTTPX_CLIENT = None
if USE_HTTPX and httpx is not None:
    try:
        HTTPX_CLIENT = httpx.Client(timeout=httpx.Timeout(connect=None, read=None), limits=httpx.Limits(max_keepalive_connections=10, max_connections=50), trust_env=True)
    except Exception:
        HTTPX_CLIENT = None

# Async httpx usage flag
USE_HTTPX_ASYNC = str(os.getenv("USE_HTTPX_ASYNC", "")).lower() in ("1", "true", "yes") and httpx is not None

import random
import requests
from pymilvus import DataType, FieldSchema, CollectionSchema, Collection, utility
from config import (
    WEB_TOP_K,
    WEB_TIMEOUT_SECONDS,
    SEARCH_RETRIES,
    BACKOFF_BASE,
    PARALLEL_SEARCH_TIMEOUT,
    WEB_BACKOFF_JITTER,
    WEB_USER_AGENTS,
    WEB_PROXY,
)

WEB_COLLECTION_PREFIX = "web_memory"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
WEB_DEBUG_LOG = "web_fallback.log"
WEB_DEBUG_META = {}

# Defensive defaults in case config import is partial or missing values
try:
    WEB_BACKOFF_JITTER  # noqa: F401
except Exception:
    WEB_BACKOFF_JITTER = 0.2

try:
    WEB_USER_AGENTS  # noqa: F401
except Exception:
    WEB_USER_AGENTS = []

try:
    WEB_PROXY  # noqa: F401
except Exception:
    WEB_PROXY = ""

# 重试与退避配置由 config.py 提供


LEGAL_KEYWORDS = {
    "法律", "民法典", "合同", "婚姻", "离婚", "继承", "侵权", "赔偿", "劳动", "仲裁", "诉讼",
    "法院", "判决", "律师", "法规", "法条", "犯罪", "刑法", "行政", "责任", "权利", "义务",
}
MEDICAL_KEYWORDS = {
    "医学", "医生", "医院", "症状", "治疗", "诊断", "药", "用药", "疾病", "血压", "高血压",
    "糖尿病", "检查", "化验", "指标", "患者", "手术", "感染", "疼痛", "发烧", "咳嗽",
}


def is_greeting_or_smalltalk(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return True
    smalltalk_keywords = ["你好", "您好", "hello", "hi", "嗨", "早上好", "下午好", "晚上好", "谢谢", "再见", "拜拜"]
    return len(text) <= 12 and any(k in text for k in smalltalk_keywords)


def is_professional_query(query: str, role: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    if role == "医生":
        return True
    keywords = LEGAL_KEYWORDS if role == "律师" else LEGAL_KEYWORDS | MEDICAL_KEYWORDS
    return any(keyword.lower() in text for keyword in keywords) or len(text) >= 12


def web_collection_name(role: str) -> str:
    suffix = "legal" if role == "律师" else "medical" if role == "医生" else "general"
    return f"{WEB_COLLECTION_PREFIX}_{suffix}"


def _strip_html(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    raw_html = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw_html)
    raw_html = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", raw_html)
    raw_html = re.sub(r"(?is)<nav.*?>.*?</nav>", " ", raw_html)
    raw_html = re.sub(r"(?is)<header.*?>.*?</header>", " ", raw_html)
    raw_html = re.sub(r"(?is)<footer.*?>.*?</footer>", " ", raw_html)
    raw_html = re.sub(r"(?is)<aside.*?>.*?</aside>", " ", raw_html)
    raw_html = re.sub(r"(?is)<form.*?>.*?</form>", " ", raw_html)
    text = re.sub(r"(?s)<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _log_debug(message: str) -> None:
    try:
        with open(WEB_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def _run_search_with_retries(engine_name: str, search_func, query: str, top_k: int) -> dict:
    """Run a search function with retries and exponential backoff. Returns dict with keys: engine, results, errors."""
    errors = []
    for attempt in range(1, SEARCH_RETRIES + 1):
        try:
            results = search_func(query, top_k)
            return {"engine": engine_name, "results": results or [], "errors": errors}
        except Exception as exc:
            tb = traceback.format_exc()
            errors.append({"attempt": attempt, "error": str(exc), "trace": tb})
            # exponential backoff with jitter
            base = BACKOFF_BASE * (2 ** (attempt - 1))
            jitter = globals().get("WEB_BACKOFF_JITTER", 0.0) * base
            sleep_time = base + random.uniform(-jitter, jitter)
            if sleep_time < 0:
                sleep_time = 0
            time.sleep(sleep_time)
            continue
    return {"engine": engine_name, "results": [], "errors": errors}


def _get_request_kwargs():
    """Return headers and proxies for requests, rotating User-Agent and applying optional proxy."""
    ua = USER_AGENT
    try:
        if isinstance(WEB_USER_AGENTS, (list, tuple)) and WEB_USER_AGENTS:
            ua = random.choice(WEB_USER_AGENTS)
        elif isinstance(WEB_USER_AGENTS, str) and WEB_USER_AGENTS:
            ua = random.choice(WEB_USER_AGENTS.split("||"))
    except Exception:
        ua = USER_AGENT
    headers = {"User-Agent": ua}
    proxies = None
    if globals().get("WEB_PROXY"):
        proxies = {"http": globals().get("WEB_PROXY"), "https": globals().get("WEB_PROXY")}
    return {"headers": headers, "proxies": proxies}


def _requests_get(url: str, headers=None, timeout=10, proxies=None):
    """Unified GET wrapper: prefer httpx client (pooling), otherwise use requests.
    Returns a SimpleNamespace with attributes: ok, text, headers, status_code
    """
    if USE_HTTPX and HTTPX_CLIENT is not None:
        try:
            kwargs = {}
            if proxies:
                kwargs["proxies"] = proxies
            resp = HTTPX_CLIENT.get(url, headers=headers, timeout=timeout, **kwargs)
            ok = 200 <= resp.status_code < 400
            return SimpleNamespace(ok=ok, text=resp.text, headers=resp.headers, status_code=resp.status_code)
        except Exception as exc:
            raise
    else:
        # fallback to requests
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        except Exception:
            raise
        status = getattr(resp, "status_code", None)
        body_preview = (resp.text or "")[:1000]
        if not (200 <= status < 400):
            _log_debug(f"requests_get non-ok url={url} status={status} preview={body_preview[:500]}")
        return SimpleNamespace(ok=getattr(resp, "ok", status < 400), text=resp.text, headers=resp.headers, status_code=status)


def _search_bing(query: str, top_k: int) -> list[dict]:
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        raise Exception(f"search engine=bing failed: status={getattr(resp, 'status_code', None)}")
    page = resp.text
    results = []
    pattern = re.compile(r'<li class="b_algo".*?<a href="(http[^"#]+)".*?>(.*?)</a>.*?(?:<p>(.*?)</p>)?', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        snippet = _strip_html(match.group(3) or "")[:300]
        if link and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": snippet, "engine": "bing"})
        if len(results) >= top_k:
            break
    return results


def _search_google(query: str, top_k: int) -> list[dict]:
    url = f"https://www.google.com/search?q={quote_plus(query)}&num={top_k}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        raise Exception(f"search engine=google failed: status={getattr(resp, 'status_code', None)}")
    page = resp.text
    results = []
    pattern = re.compile(r'/url\?q=(https?://[^&"#]+).*?<h3.*?>(.*?)</h3>', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        if link and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": "", "engine": "google"})
        if len(results) >= top_k:
            break
    return results


def _search_baidu(query: str, top_k: int) -> list[dict]:
    url = f"https://www.baidu.com/s?wd={quote_plus(query)}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        # Treat as zero results rather than raising to allow other engines to proceed
        _log_debug(f"search engine=baidu failed status={getattr(resp,'status_code',None)}")
        return []
    page = resp.text
    results = []
    pattern = re.compile(r'<h3[^>]*class="[^"]*c-title[^"]*".*?<a[^>]+href="([^"]+)".*?>(.*?)</a>', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        if link.startswith("/"):
            link = urljoin("https://www.baidu.com", link)
        if link and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": "", "engine": "baidu"})
        if len(results) >= top_k:
            break
    return results


def _search_ddg(query: str, top_k: int) -> list[dict]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        raise Exception(f"search engine=ddg failed: status={getattr(resp, 'status_code', None)}")
    page = resp.text
    results = []
    pattern = re.compile(r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        if link and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": "", "engine": "ddg"})
        if len(results) >= top_k:
            break
    return results


def _search_360(query: str, top_k: int) -> list[dict]:
    url = f"https://www.so.com/s?q={quote_plus(query)}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        _log_debug(f"search engine=360 failed status={getattr(resp,'status_code',None)}")
        return []
    page = resp.text
    results = []
    pattern = re.compile(r'<h3[^>]*>.*?<a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        if link and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": "", "engine": "360"})
        if len(results) >= top_k:
            break
    return results


def _search_sogou(query: str, top_k: int) -> list[dict]:
    url = f"https://www.sogou.com/web?query={quote_plus(query)}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        _log_debug(f"search engine=sogou failed status={getattr(resp,'status_code',None)}")
        return []
    page = resp.text
    results = []
    pattern = re.compile(r'<a[^>]+href="(https?://[^"\s]+)"[^>]*class="?vr-title"?[^>]*>.*?<em.*?>.*?</em>(.*?)</a>', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        if link and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": "", "engine": "sogou"})
        if len(results) >= top_k:
            break
    return results


def _search_shenma(query: str, top_k: int) -> list[dict]:
    url = f"https://www.sm.cn/s?q={quote_plus(query)}"
    kwargs = _get_request_kwargs()
    resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
    if not resp.ok:
        _log_debug(f"search engine=shenma failed status={getattr(resp,'status_code',None)}")
        return []
    page = resp.text
    results = []
    pattern = re.compile(r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>', re.S)
    for match in pattern.finditer(page):
        link = html.unescape(match.group(1))
        title = _strip_html(match.group(2))[:120]
        if link and ("http" in link) and not any(item["url"] == link for item in results):
            results.append({"title": title, "url": link, "snippet": "", "engine": "shenma"})
        if len(results) >= top_k:
            break
    return results


def _fetch_page(url: str) -> str:
    try:
        kwargs = _get_request_kwargs()
        resp = _requests_get(url, headers=kwargs.get("headers"), timeout=WEB_TIMEOUT_SECONDS, proxies=kwargs.get("proxies"))
        status = getattr(resp, "status_code", None)
        content_type = resp.headers.get("content-type", "") if resp.headers is not None else ""
        text = ""
        if resp.ok and ("text/html" in content_type or "text/plain" in content_type):
            text = _strip_html(resp.text)
        length = len(text) if text else 0
        _log_debug(f"fetch_page url={url} status={status} length={length} content_type={content_type}")
        return text[:4000]
    except Exception as exc:
        tb = traceback.format_exc()
        _log_debug(f"fetch_page_failed url={url} error={exc} trace={tb}")
        return ""


def _extract_core_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？.!?])\s*", text)
    blacklist = ["登录", "注册", "隐私", "版权", "广告", "导航", "首页", "联系我们", "关于我们", "返回顶部", "下一页", "上一页", "推荐阅读", "猜你喜欢"]
    useful = []
    for part in parts:
        p = part.strip()
        if len(p) < 8:
            continue
        if any(bad in p for bad in blacklist):
            continue
        useful.append(p)
    return useful


def _split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> Iterable[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += max(1, chunk_size - overlap)
    return chunks


def crawl_web_knowledge(query: str, role: str, top_k: int = WEB_TOP_K) -> list[dict]:
    if not query or is_greeting_or_smalltalk(query):
        return []

    search_query = query
    if role == "律师":
        search_query = f"{query} 法律 法规 解释"
    elif role == "医生":
        search_query = f"{query} 医学 指南 健康"

    _log_debug(f"start crawl role={role} query={query} search_query={search_query}")

    # 如果启用了异步 httpx 路径，优先走 async 实现
    if USE_HTTPX_ASYNC and httpx is not None:
        try:
            import asyncio

            async def _run_async():
                return await async_crawl_web_knowledge(query, role, top_k=top_k)

            return asyncio.run(_run_async())
        except Exception:
            # 回退到同步实现
            pass

    # 支持通过环境变量 `WEB_ENGINES` 指定搜索引擎及顺序，例如: "baidu,bing,google"
    env_engines = os.getenv("WEB_ENGINES", "").strip()
    default_engines = [
        ("ddg", _search_ddg),
        ("google", _search_google),
        ("baidu", _search_baidu),
        ("bing", _search_bing),
        ("360", _search_360),
        ("sogou", _search_sogou),
        ("shenma", _search_shenma),
    ]
    if env_engines:
        names = [n.strip().lower() for n in env_engines.split(",") if n.strip()]
        # 保持声明中存在的引擎顺序并过滤无效项
        mapping = {"ddg": _search_ddg, "google": _search_google, "baidu": _search_baidu, "bing": _search_bing}
        search_engine_list = [(n, mapping[n]) for n in names if n in mapping]
        if not search_engine_list:
            search_engine_list = default_engines
    else:
        search_engine_list = default_engines

    # 并行触发各搜索引擎并带重试/退避策略
    futures = {}
    aggregated = []
    matched_engine = ""
    search_results = []
    errors_by_engine = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_engine_list)) as ex:
        for engine_name, search_func in search_engine_list:
            fut = ex.submit(_run_search_with_retries, engine_name, search_func, search_query, top_k)
            futures[fut] = engine_name

        # 等待指定超时时间，收集已完成的 futures；超时后继续处理已完成结果
        done, not_done = concurrent.futures.wait(list(futures.keys()), timeout=PARALLEL_SEARCH_TIMEOUT)

        for fut in done:
            try:
                res = fut.result()
            except Exception as exc:
                tb = traceback.format_exc()
                engine = futures.get(fut, "unknown")
                errors_by_engine[engine] = errors_by_engine.get(engine, []) + [{"error": str(exc), "trace": tb}]
                continue
            engine = res.get("engine")
            errors = res.get("errors") or []
            if errors:
                errors_by_engine[engine] = errors_by_engine.get(engine, []) + errors
            results = res.get("results") or []
            aggregated.append((engine, results))

        # 记录未完成的 futures（便于排查和降级策略）
        if not_done:
            pending_engines = [futures.get(fut, "unknown") for fut in not_done]
            _log_debug(f"search_timeout_pending_engines={pending_engines}")
            WEB_DEBUG_META["pending_engines"] = pending_engines

        # 按优先级顺序选择第一个非空结果集（来自已完成的引擎）
        for engine_name, _ in search_engine_list:
            for eng, results in aggregated:
                if eng == engine_name and results:
                    search_results = results
                    matched_engine = eng
                    break
            if matched_engine:
                break

        # 记录所有引擎的错误信息，便于排查
        if errors_by_engine:
            _log_debug(f"search_errors: {errors_by_engine}")
            WEB_DEBUG_META["errors_by_engine"] = errors_by_engine

    if not search_results:
        _log_debug("all search engines returned no results")
        return []

    documents = []
    for result in search_results:
        text = ""
        try:
            text = _fetch_page(result["url"])
        except Exception:
            text = ""
        if text:
            _log_debug(f"fetch page url={result['url']} text_len={len(text)}")
        else:
            text = result.get("snippet", "")
            if text:
                _log_debug(f"fallback to snippet url={result['url']} snippet_len={len(text)}")

        core_sentences = _extract_core_sentences(text)
        if core_sentences:
            # 只保留标题 + 正文核心段落 + 适合问答的知识点
            knowledge_points = []
            for sent in core_sentences[:8]:
                if len(sent) >= 12:
                    knowledge_points.append(sent)
            compact_text = "\n".join(knowledge_points[:6]).strip() or text
        else:
            compact_text = text

        chunks = list(_split_text(compact_text))
        if not chunks:
            _log_debug(f"no chunks after split url={result['url']}")
            if compact_text:
                doc_id = hashlib.sha1(f"{role}|{result['url']}|snippet|{compact_text[:80]}".encode("utf-8")).hexdigest()
                documents.append({
                    "id": doc_id,
                    "title": result.get("title") or "互联网资料",
                    "content": compact_text,
                    "page": 1,
                    "source_file": result["url"],
                    "url": result["url"],
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "engine": result.get("engine", matched_engine or "unknown"),
                })
                continue
        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.sha1(f"{role}|{result['url']}|{idx}|{chunk[:80]}".encode("utf-8")).hexdigest()
            documents.append({
                "id": doc_id,
                "title": result.get("title") or "互联网资料",
                "content": chunk,
                "page": idx + 1,
                "source_file": result["url"],
                "url": result["url"],
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "engine": result.get("engine", matched_engine or "unknown"),
            })

    WEB_DEBUG_META.update({
        "query": query,
        "role": role,
        "matched_engine": matched_engine or "unknown",
        "document_count": len(documents),
        "timestamp": datetime.now().isoformat(),
    })
    _log_debug(f"crawl completed documents={len(documents)} engine={matched_engine or 'unknown'}")
    return documents


def get_last_web_debug() -> dict:
    """Return a shallow copy of the last web debug meta information."""
    try:
        return dict(WEB_DEBUG_META)
    except Exception:
        return {}


async def async_crawl_web_knowledge(query: str, role: str, top_k: int = WEB_TOP_K) -> list[dict]:
    """Async version of crawl_web_knowledge using httpx.AsyncClient. Falls back to sync helpers for parsing.
    Note: this requires httpx installed and USE_HTTPX_ASYNC=1 to be enabled.
    """
    if httpx is None:
        return crawl_web_knowledge(query, role, top_k=top_k)

    search_query = query
    if role == "律师":
        search_query = f"{query} 法律 法规 解释"
    elif role == "医生":
        search_query = f"{query} 医学 指南 健康"

    _log_debug(f"start async crawl role={role} query={query} search_query={search_query}")

    default_engines = [
        ("ddg", _search_ddg),
        ("google", _search_google),
        ("baidu", _search_baidu),
        ("bing", _search_bing),
    ]
    env_engines = os.getenv("WEB_ENGINES", "").strip()
    if env_engines:
        names = [n.strip().lower() for n in env_engines.split(",") if n.strip()]
        mapping = {"ddg": _search_ddg, "google": _search_google, "baidu": _search_baidu, "bing": _search_bing}
        search_engine_list = [(n, mapping[n]) for n in names if n in mapping]
        if not search_engine_list:
            search_engine_list = default_engines
    else:
        search_engine_list = default_engines

    aggregated = []
    matched_engine = ""
    search_results = []
    errors_by_engine = {}

    timeout = float(PARALLEL_SEARCH_TIMEOUT)

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=None, read=None), limits=httpx.Limits(max_keepalive_connections=10, max_connections=50), trust_env=True) as client:
        # run searches sequentially async to avoid complex parallelism here
        for engine_name, search_func in search_engine_list:
            try:
                # call the sync search parser but use client to fetch pages when needed
                # We reuse existing sync search functions which perform parsing on returned HTML
                # To integrate, temporarily swap _requests_get to use client
                original_requests_get = globals().get("_requests_get")

                async def _async_requests_get(url: str, headers=None, timeout=WEB_TIMEOUT_SECONDS, proxies=None):
                    try:
                        resp = await client.get(url, headers=headers, timeout=timeout)
                        ok = 200 <= resp.status_code < 400
                        return SimpleNamespace(ok=ok, text=resp.text, headers=resp.headers, status_code=resp.status_code)
                    except Exception as exc:
                        raise

                globals()["_requests_get"] = _async_requests_get  # type: ignore
                # call sync wrapper which will now call the async-compatible _requests_get
                res = await asyncio.to_thread(_run_search_with_retries, engine_name, search_func, search_query, top_k)
                globals()["_requests_get"] = original_requests_get
            except Exception as exc:
                tb = traceback.format_exc()
                errors_by_engine[engine_name] = errors_by_engine.get(engine_name, []) + [{"error": str(exc), "trace": tb}]
                continue
            engine = res.get("engine")
            errors = res.get("errors") or []
            if errors:
                errors_by_engine[engine] = errors_by_engine.get(engine, []) + errors
            results = res.get("results") or []
            aggregated.append((engine, results))

    for engine_name, _ in search_engine_list:
        for eng, results in aggregated:
            if eng == engine_name and results:
                search_results = results
                matched_engine = eng
                break
        if matched_engine:
            break

    if errors_by_engine:
        WEB_DEBUG_META["errors_by_engine"] = errors_by_engine

    documents = []
    for result in search_results:
        text = ""
        try:
            # fetch page with httpx AsyncClient
            async with httpx.AsyncClient(timeout=WEB_TIMEOUT_SECONDS, trust_env=True) as ac:
                kwargs = _get_request_kwargs()
                r = await ac.get(result["url"], headers=kwargs.get("headers"))
                if 200 <= r.status_code < 400:
                    text = _strip_html(r.text)
        except Exception:
            text = result.get("snippet", "")

        core_sentences = _extract_core_sentences(text)
        if core_sentences:
            knowledge_points = []
            for sent in core_sentences[:8]:
                if len(sent) >= 12:
                    knowledge_points.append(sent)
            compact_text = "\n".join(knowledge_points[:6]).strip() or text
        else:
            compact_text = text

        chunks = list(_split_text(compact_text))
        if not chunks and compact_text:
            doc_id = hashlib.sha1(f"{role}|{result['url']}|snippet|{compact_text[:80]}".encode("utf-8")).hexdigest()
            documents.append({
                "id": doc_id,
                "title": result.get("title") or "互联网资料",
                "content": compact_text,
                "page": 1,
                "source_file": result["url"],
                "url": result["url"],
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "engine": result.get("engine", matched_engine or "unknown"),
            })
            continue
        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.sha1(f"{role}|{result['url']}|{idx}|{chunk[:80]}".encode("utf-8")).hexdigest()
            documents.append({
                "id": doc_id,
                "title": result.get("title") or "互联网资料",
                "content": chunk,
                "page": idx + 1,
                "source_file": result["url"],
                "url": result["url"],
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "engine": result.get("engine", matched_engine or "unknown"),
            })

    WEB_DEBUG_META.update({
        "query": query,
        "role": role,
        "matched_engine": matched_engine or "unknown",
        "document_count": len(documents),
        "timestamp": datetime.now().isoformat(),
    })
    return documents


def ensure_web_collection(collection_name: str, dim: int) -> Collection:
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="fetched_at", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="互联网兜底知识长期记忆")
    collection = Collection(collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
    )
    return collection


def save_web_knowledge_to_milvus(documents: list[dict], model, collection_name: str) -> int:
    if not documents:
        return 0
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts, normalize_embeddings=True)
    dim = len(embeddings[0])
    collection = ensure_web_collection(collection_name, dim)
    rows = []
    for doc, embedding in zip(documents, embeddings):
        rows.append({
            "id": doc["id"],
            "title": doc["title"][:512],
            "content": doc["content"][:8192],
            "page": int(doc.get("page") or 0),
            "source_file": doc.get("source_file", "")[:1024],
            "url": doc.get("url", "")[:1024],
            "fetched_at": doc.get("fetched_at", "")[:64],
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
        })
    collection.insert(rows)
    collection.flush()
    try:
        collection.load()
    except Exception:
        time.sleep(0.2)
        collection.load()
    return len(rows)


def build_web_context(documents: list[dict], limit: int = 5) -> tuple[str, list[dict]]:
    context_parts = []
    refs = []
    seen = set()
    for idx, doc in enumerate(documents[:limit], start=1):
        dedupe_key = f"{doc.get('source_file', '')}|{_strip_html(doc.get('title', '')).lower()}|{_strip_html(doc.get('content', '')).lower()[:240]}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        short_content = _strip_html(doc.get('content', ''))[:1000]
        context_parts.append(
            f"[联网资料{idx}] 来源：{doc.get('source_file', '')}\n"
            f"标题：{doc.get('title', '')}\n"
            f"抓取时间：{doc.get('fetched_at', '')}\n"
            f"内容：{short_content}"
        )
        refs.append({
            "rank": idx,
            "id": doc.get("id"),
            "title": doc.get("title"),
            "page": doc.get("page"),
            "source_file": doc.get("source_file"),
            "base_score": 0.0,
            "rerank_score": 0.0,
            "source": "web",
        })
    return "\n\n".join(context_parts), refs
