import json
from pathlib import Path
from src.retrieval import vector_retrieval, get_parent_pages, llm_reranking
from src.prompts import (
    AnswerWithRAGContextBooleanPrompt,
    AnswerWithRAGContextNumberPrompt,
    AnswerWithRAGContextStringPrompt
)
from dashscope import Generation
from dotenv import load_dotenv
import os

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

def extract_company_name(query: str) -> str:
    """从查询中提取公司名（示例：中芯国际）"""
    # 简化实现：假设查询含"中芯国际"，实际可用LLM提取
    if "中芯国际" in query:
        return "中芯国际"
    raise Exception(f"未从查询中提取到公司名：{query}")

def build_context(reranked_pages: list[dict]) -> str:
    """构建上下文（带页码）"""
    context_parts = []
    for page in reranked_pages:
        context_parts.append(f"页码{page['page_num']}：{page['full_text'][:1500]}")  # 截取前1500字符
    return "\n\n".join(context_parts)

def generate_answer(query: str, kind: str, db_dir: Path, md_dir: Path) -> dict:
    """生成答案（按kind路由到对应Prompt）"""
    # 1. 提取公司名
    company_name = extract_company_name(query)
    # 2. 检索与重排序
    retrieved_chunks = vector_retrieval(query, company_name, db_dir)
    parent_pages = get_parent_pages(retrieved_chunks, md_dir)
    reranked_pages = llm_reranking(query, parent_pages)
    # 3. 构建上下文
    context = build_context(reranked_pages)
    # 4. 按kind选择Prompt
    if kind == "boolean":
        prompt_cls = AnswerWithRAGContextBooleanPrompt
    elif kind == "number":
        prompt_cls = AnswerWithRAGContextNumberPrompt
    elif kind == "string":
        prompt_cls = AnswerWithRAGContextStringPrompt
    else:
        raise Exception(f"不支持的问题类型：{kind}")
    # 5. 构造Prompt
    user_prompt = prompt_cls.user_prompt.format(context=context, question=query)
    full_prompt = f"{prompt_cls.instruction}\n\n{user_prompt}\n\n请按{prompt_cls.AnswerSchema.__name__}格式输出JSON"
    # 6. 调用LLM生成答案
    response = Generation.call(
        "qwen-turbo-latest",
        messages=[{"role": "user", "content": full_prompt}],
        api_key=DASHSCOPE_API_KEY
    )
    return json.loads(response.output.choices[0].message.content)