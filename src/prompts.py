from pydantic import BaseModel, Field, Literal
from typing import List, Union

# 1. 共享系统指令
class AnswerWithRAGContextSharedPrompt:
    instruction = """你是RAG问答系统，仅基于公司年报/投研报告的检索内容回答，需标注引用页码，不编造信息。"""
    user_prompt = """
    以下是上下文（带页码）：
    {context}
    以下是问题：
    {question}
    """

# 2. Boolean类型（是/否）
class AnswerWithRAGContextBooleanPrompt(AnswerWithRAGContextSharedPrompt):
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="至少5步推理，150字以上，避免被相似信息迷惑")
        reasoning_summary: str = Field(description="约50字推理总结")
        relevant_pages: List[int] = Field(description="仅包含直接支持答案的页码，至少1个")
        final_answer: bool = Field(description="True/False，未发生则返回False")

# 3. Number类型（数值）
class AnswerWithRAGContextNumberPrompt(AnswerWithRAGContextSharedPrompt):
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="至少5步推理，严格匹配指标定义，不接受代理指标或计算推导")
        reasoning_summary: str = Field(description="约50字推理总结")
        relevant_pages: List[int] = Field(description="仅包含直接支持答案的页码，至少1个")
        final_answer: Union[float, int, Literal["N/A"]] = Field(description="数值或'N/A'（无匹配指标时）")

# 4. String类型（开放性文本，新增）
class AnswerWithRAGContextStringPrompt(AnswerWithRAGContextSharedPrompt):
    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="至少5步推理，结合上下文归纳，150字以上")
        reasoning_summary: str = Field(description="约50字推理总结")
        relevant_pages: List[int] = Field(description="仅包含直接支持答案的页码，至少1个")
        final_answer: str = Field(description="完整连贯的文本，无信息时说明'未找到答案'")