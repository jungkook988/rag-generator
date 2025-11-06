import streamlit as st
from pathlib import Path
from src.pipeline import Pipeline

# 初始化Pipeline
root_path = Path("data")
pipeline = Pipeline(root_path)

# 页面配置
st.set_page_config(page_title="企业知识库RAG系统", page_icon="📊")
st.markdown("""
    <div style="background:linear-gradient(90deg,#7b2ff2 28%,#f22f7b);padding:20px;border-radius:10px;">
        <h2 style="color:white;margin:0;">企业知识库RAG系统</h2>
        <p style="color:white;font-size:16px;">基于中芯国际年报与投研报告的问答系统</p>
    </div>
""", unsafe_allow_html=True)

# 侧边栏：预处理触发与问题设置
with st.sidebar:
    st.subheader("系统设置")
    if st.button("执行预处理（PDF→Faiss库）"):
        with st.spinner("预处理中...（约5-10分钟）"):
            pipeline.run_preprocessing()
        st.success("预处理完成！")
    st.subheader("问题输入")
    query = st.text_input("请输入你的问题：")
    kind = st.selectbox("问题类型：", ["string", "number", "boolean"])
    generate_btn = st.button("生成答案")

# 主区域：显示答案
if generate_btn and query:
    with st.spinner("检索与生成答案中..."):
        try:
            answer = pipeline.answer_single_question(query, kind)
            # 显示结果
            st.subheader("检索结果与答案")
            st.write(f"**分步推理：** {answer['step_by_step_analysis']}")
            st.write(f"**推理摘要：** {answer['reasoning_summary']}")
            st.write(f"**相关页码：** {answer['relevant_pages']}")
            st.write(f"**最终答案：** {answer['final_answer']}")
        except Exception as e:
            st.error(f"生成答案出错：{str(e)}")
else:
    st.info("请在左侧输入问题并点击【生成答案】")