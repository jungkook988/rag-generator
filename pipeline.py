# src/pipeline.py（适配Chroma）
from pathlib import Path
from src.pdf_parsing import parse_pdf_with_docling
from src.text_splitter import build_chunked_reports
from src.ingestion import build_chroma_db  # 替换原build_faiss_db
from src.generation import generate_answer

class Pipeline:
    def __init__(self, root_path: Path = Path("data/stock_data")):
        self.root_path = root_path
        self.pdf_dir = root_path / "pdf_reports"
        self.json_dir = root_path / "parsed_json"
        self.md_dir = root_path / "parsed_md"
        self.chunk_dir = root_path / "databases/chunked_reports"
        # 向量库路径改为Chroma的存储目录（无需手动创建，Chroma自动管理）

    def run_preprocessing(self):
        """预处理步骤：仅替换向量库构建部分"""
        # 1. PDF解析为JSON（不变）
        self.json_dir.mkdir(parents=True, exist_ok=True)
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            output_json = self.json_dir / f"{pdf_file.stem}.json"
            parse_pdf_with_docling(pdf_file, output_json)
        # 2. JSON转MD（不变）
        self.md_dir.mkdir(parents=True, exist_ok=True)
        for json_file in self.json_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                pages_data = json.load(f)
            md_content = []
            for page in pages_data:
                md_content.append(f"# 第{page['page_num']}页")
                md_content.append(page['text'])
                for table in page['tables']:
                    md_content.append("| " + " | ".join(table["headers"]) + " |")
                    md_content.append("| " + " | ".join(["---"]*len(table["headers"])) + " |")
                    for row in table["rows"]:
                        md_content.append("| " + " | ".join(row) + " |")
            with open(self.md_dir / f"{json_file.stem}.md", "w", encoding="utf-8") as f:
                f.write("\n\n".join(md_content))
        # 3. 文本分块（不变）
        build_chunked_reports(self.md_dir, self.chunk_dir)
        # 4. 构建Chroma向量库（替换原Faiss步骤）
        build_chroma_db(self.chunk_dir)
        print("预处理完成！")

    def answer_single_question(self, query: str, kind: str) -> dict:
        """回答逻辑不变"""
        return generate_answer(query, kind, self.db_dir, self.md_dir)  # 此处db_dir实际已被Chroma内部管理