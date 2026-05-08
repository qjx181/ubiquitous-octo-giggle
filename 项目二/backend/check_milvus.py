from pymilvus import MilvusClient

c = MilvusClient("http://localhost:19530")

# 尝试 query_count (新版API)
try:
    n = c.query_count(collection_name="prospectus")
    print("query_count:", n)
except:
    pass

# 直接查询所有数据，看有多少条
r = c.query(collection_name="prospectus", filter="", output_fields=["chunk_id"], limit=10)
print("前10条chunk_id:", [x["chunk_id"] for x in r])

# 用 count 方式
try:
    r2 = c.query(collection_name="prospectus", filter="chunk_id != ''", output_fields=["chunk_id"], limit=1)
    print("第一条chunk_id:", r2[0]["chunk_id"] if r2 else "无数据")
except Exception as e:
    print("查询失败:", e)
