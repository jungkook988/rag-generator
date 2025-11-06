from pathlib import Path
import re
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("text-embedding-3-large")

def split_markdown_file(md_path: Path, chunk_size: int = 300, chunk_overlap: int = 20) -> list[dict]:
    """按300token分块，记录父页面编号"""
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    # 按页面分割（假设MD中用"# 第X页"分隔）
    page_sections = re.split(r"# 第(\d+)页", md_content)[1:]  # 提取（页码，内容）
    chunks = []
    for i in range(0, len(page_sections), 2):
        page_num = int(page_sections[i])
        page_text = page_sections[i+1]
        # 按token分块
        tokens = tokenizer.encode(page_text)
        for j in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[j:j+chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append({
                "chunk_id": f"{md_path.stem}_page{page_num}_chunk{j//(chunk_size - chunk_overlap)}",
                "parent_page": page_num,
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })
    return chunks

def build_chunked_reports(md_dir: Path, output_dir: Path):
    """批量分块并保存到stock_data/databases/chunked_reports"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for md_file in md_dir.glob("*.md"):
        chunks = split_markdown_file(md_file)
        # 保存分块结果
        with open(output_dir / f"{md_file.stem}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)