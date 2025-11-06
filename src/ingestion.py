import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import os

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-large"
model = SentenceTransformer(EMBEDDING_MODEL)

# 初始化Chroma客户端（本地存储）
chroma_client = chromadb.Client(
    Settings(
        persist_directory=str(Path("data/stock_data/databases/chroma_db")),  # 向量数据存储路径
        anonymized_telemetry=False  # 关闭匿名统计
    )
)

def build_chroma_db(chunk_dir: Path):
    """为每个公司的分块构建Chroma集合（替代Faiss库）"""
    # 遍历所有分块文件（如"中芯国际_chunks.json"）
    for chunk_file in chunk_dir.glob("*_chunks.json"):
        company_name = chunk_file.stem.replace("_chunks", "")
        # 1. 创建/获取公司专属集合（Collection）
        collection = chroma_client.get_or_create_collection(
            name=company_name,  # 集合名=公司名，实现分库
            metadata={"description": f"{company_name}的年报/投研报告分块向量"}
        )
        # 2. 读取分块数据
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        # 3. 生成向量与ID
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(chunk_texts, convert_to_numpy=True).tolist()  # 转为列表（Chroma要求）
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]  # 用chunk_id作为唯一标识
        # 4. 向集合中添加向量+文本+元数据（替代Faiss的索引+metadata.json）
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=[  # 存储父页面、token数等元数据
                {
                    "parent_page": chunk["parent_page"],
                    "token_count": chunk["token_count"]
                } for chunk in chunks
            ]
        )
        # 持久化到本地
        chroma_client.persist()
        print(f"Chroma集合构建完成：{company_name}（向量数：{len(embeddings)}）")