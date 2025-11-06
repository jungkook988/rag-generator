# src/retrieval.py（Chroma版本）
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dashscope import Generation
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
import os
import re

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
model = SentenceTransformer(EMBEDDING_MODEL)

# 初始化Chroma客户端（与构建时路径一致）
chroma_client = Client(
    Settings(
        persist_directory=str(Path("data/stock_data/databases/chroma_db")),
        anonymized_telemetry=False
    )
)

def vector_retrieval(query: str, company_name: str, top_k: int = 30) -> list[dict]:
    """用Chroma检索Top30分块（替代Faiss的vector_retrieval）"""
    # 1. 获取公司专属集合
    try:
        collection = chroma_client.get_collection(name=company_name)
    except ValueError:
        raise Exception(f"未找到{company_name}的Chroma集合，请先执行预处理")
    # 2. query向量化
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    # 3. 检索Top30（按内积相似度，与Faiss的IndexFlatIP一致）
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # 返回文本、元数据、距离
    )
    # 4. 整理结果（与原Faiss输出格式对齐，便于后续处理）
    retrieved_chunks = []
    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "parent_page": results["metadatas"][0][i]["parent_page"],
            "token_count": results["metadatas"][0][i]["token_count"],
            "similarity_score": 1 - results["distances"][0][i]  # 内积越大越相似，转为0-1得分
        })
    return retrieved_chunks

# 以下函数（get_parent_pages、llm_reranking）与原版本完全一致，无需修改
def get_parent_pages(retrieved_chunks: list[dict], md_dir: Path) -> list[dict]:
    """通过分块元数据定位父页面内容（复用原逻辑）"""
    parent_pages = {}
    page_nums = list(set([chunk["parent_page"] for chunk in retrieved_chunks]))
    company_name = retrieved_chunks[0]["chunk_id"].split("_page")[0]
    md_path = md_dir / f"{company_name}.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    page_sections = re.split(r"# 第(\d+)页", md_content)[1:]
    for i in range(0, len(page_sections), 2):
        page_num = int(page_sections[i])
        if page_num in page_nums:
            parent_pages[page_num] = {
                "page_num": page_num,
                "full_text": page_sections[i+1],
                "source_chunk_ids": [c["chunk_id"] for c in retrieved_chunks if c["parent_page"] == page_num]
            }
    return list(parent_pages.values())

def llm_reranking(query: str, parent_pages: list[dict], retrieved_chunks=None) -> list[dict]:
    """LLM重排序（复用原逻辑）"""
    reranked_pages = []
    for page in parent_pages:
        prompt = f"""
        你是RAG检索重排专家，请根据查询与页面内容的相关性评分（0-1，步长0.1）。
        查询：{query}
        页面内容：{page['full_text'][:1000]}
        输出格式：{{"reasoning": "分析理由", "relevance_score": 0.0}}
        """
        response = Generation.call(
            "qwen-turbo-latest",
            messages=[{"role": "user", "content": prompt}],
            api_key=DASHSCOPE_API_KEY
        )
        result = json.loads(response.output.choices[0].message.content)
        vector_score = np.mean([c["similarity_score"] for c in retrieved_chunks if c["parent_page"] == page["page_num"]])
        weighted_score = (vector_score * 0.3) + (result["relevance_score"] * 0.7)
        reranked_pages.append({
            **page,
            "llm_reasoning": result["reasoning"],
            "llm_score": result["relevance_score"],
            "weighted_score": round(weighted_score, 2)
        })
    return sorted(reranked_pages, key=lambda x: x["weighted_score"], reverse=True)[:10]