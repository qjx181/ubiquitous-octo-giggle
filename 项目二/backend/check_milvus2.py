from pymilvus import MilvusClient
c = MilvusClient("http://localhost:19530")
desc = c.describe_collection("prospectus")
print("describe_collection 完整内容:")
for k, v in desc.items():
    print(f"  {k}: {v}")
