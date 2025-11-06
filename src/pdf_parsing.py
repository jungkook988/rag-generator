import requests
import time
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
MINERU_API_KEY = os.getenv("MINERU_API_KEY")

def get_task_id(file_name: str) -> str:
    """获取MinerU解析任务ID"""
    url = "https://mineru.net/api/v4/extract/task"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_KEY}"
    }
    # 若本地文件，需先上传至OSS获取URL，此处用示例URL
    pdf_url = f"https://your-oss-url/pdf/{file_name}"
    data = {"url": pdf_url, "is_ocr": True, "enable_formula": False}
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"]["task_id"]

def get_mineru_result(task_id: str, output_dir: Path):
    """获取解析结果并解压"""
    url = f"https://mineru.net/api/v4/extract/result/{task_id}"
    headers = {"Authorization": f"Bearer {MINERU_API_KEY}"}
    while True:
        response = requests.get(url, headers=headers)
        result = response.json()
        if result["state"] == "done":
            # 下载ZIP并解压
            zip_url = result["data"]["full_zip_url"]
            zip_path = output_dir / f"{task_id}.zip"
            with requests.get(zip_url, stream=True) as r:
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            # 解压
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(output_dir / task_id)
            print(f"解析完成：{output_dir / task_id}")
            break
        elif result["state"] == "error":
            raise Exception(f"MinerU解析错误：{result['err_msg']}")
        time.sleep(5)  # 轮询等待