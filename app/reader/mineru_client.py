from pathlib import Path
import io
import os
import time
import zipfile

import requests
from dotenv import load_dotenv


class MinerUClient:
    """MinerU 精准解析 API 客户端。

    该类只负责与 MinerU 云端 API 通信，不负责创建 Document 对象。

    流程：
        申请上传 URL -> 上传文件 -> 轮询解析结果 -> 下载 ZIP -> 提取 full.md
    """

    def __init__(self) -> None:
        """初始化 MinerUClient，并从环境变量读取 API Token。

        Raises:
            RuntimeError: 当环境变量 MINERU_API_TOKEN 不存在时抛出。
        """
        load_dotenv()
        token = os.getenv("MINERU_API_TOKEN")

        if not token:
            raise RuntimeError("Missing MINERU_API_TOKEN in .env")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def parse_file(
        self,
        path: Path,
        *,
        language: str = "ch",
        model_version: str = "vlm",
        is_ocr: bool = True,
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> str:
        """将本地文件上传至 MinerU，并返回解析后的 Markdown 文本。

        Args:
            path (Path): 本地文件路径。
            language (str): 文档语言，例如 "ch"。
            model_version (str): MinerU 模型版本，例如 "vlm"。
            is_ocr (bool): 是否启用 OCR。
            enable_formula (bool): 是否启用公式识别。
            enable_table (bool): 是否启用表格识别。

        Returns:
            str: MinerU 解析得到的 Markdown 文本。

        Raises:
            RuntimeError: 当上传、查询或解析失败时抛出。
            TimeoutError: 当解析任务超时时抛出。
        """
        batch_id, upload_url = self._apply_upload_url(
            path=path,
            language=language,
            model_version=model_version,
            is_ocr=is_ocr,
            enable_formula=enable_formula,
            enable_table=enable_table,
        )

        self._upload_file(path, upload_url)
        full_zip_url = self._poll_result(batch_id)

        return self._download_markdown(full_zip_url)

    def _apply_upload_url(
        self,
        path: Path,
        *,
        language: str,
        model_version: str,
        is_ocr: bool,
        enable_formula: bool,
        enable_table: bool,
    ) -> tuple[str, str]:
        """申请 MinerU 文件上传 URL。

        Args:
            path (Path): 本地文件路径。
            language (str): 文档语言。
            model_version (str): MinerU 模型版本。
            is_ocr (bool): 是否启用 OCR。
            enable_formula (bool): 是否启用公式识别。
            enable_table (bool): 是否启用表格识别。

        Returns:
            tuple[str, str]: batch_id 与 upload_url。
        """
        apply_url = "https://mineru.net/api/v4/file-urls/batch"

        payload = {
            "files": [
                {
                    "name": path.name,
                    "data_id": path.stem,
                    "is_ocr": is_ocr,
                }
            ],
            "model_version": model_version,
            "language": language,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
        }

        response = requests.post(
            apply_url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU apply upload url failed: {result}")

        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]

        return batch_id, upload_url

    def _upload_file(self, path: Path, upload_url: str) -> None:
        """上传本地文件到 MinerU 提供的临时上传地址。

        Args:
            path (Path): 本地文件路径。
            upload_url (str): MinerU 返回的上传地址。

        Raises:
            RuntimeError: 当文件上传失败时抛出。
        """
        with open(path, "rb") as file:
            response = requests.put(upload_url, data=file, timeout=300)

        if response.status_code != 200:
            raise RuntimeError(
                f"MinerU file upload failed: {response.status_code}, {response.text}"
            )

    def _poll_result(self, batch_id: str) -> str:
        """轮询 MinerU 解析结果。

        Args:
            batch_id (str): MinerU 批处理任务 ID。

        Returns:
            str: 解析结果 ZIP 文件下载地址。

        Raises:
            RuntimeError: 当查询失败或解析失败时抛出。
            TimeoutError: 当解析任务超时时抛出。
        """
        query_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"

        for _ in range(120):
            response = requests.get(query_url, headers=self.headers, timeout=60)
            response.raise_for_status()

            result = response.json()
            if result.get("code") != 0:
                raise RuntimeError(f"MinerU query failed: {result}")

            extract_results = result["data"].get("extract_result", [])
            if not extract_results:
                time.sleep(5)
                continue

            item = extract_results[0]
            state = item.get("state")

            if state == "done":
                return item["full_zip_url"]

            if state == "failed":
                raise RuntimeError(f"MinerU parse failed: {item.get('err_msg')}")

            print(f"MinerU parsing state: {state}")
            time.sleep(5)

        raise TimeoutError("MinerU parse timeout")

    def _download_markdown(self, full_zip_url: str) -> str:
        """下载 MinerU 结果 ZIP，并提取 full.md。

        Args:
            full_zip_url (str): MinerU 返回的完整结果 ZIP 下载地址。

        Returns:
            str: full.md 中的 Markdown 文本。

        Raises:
            RuntimeError: 当 ZIP 中不存在 full.md 时抛出。
        """
        response = requests.get(full_zip_url, timeout=300)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            markdown_files = [
                name for name in zip_file.namelist() if name.endswith("full.md")
            ]

            if not markdown_files:
                raise RuntimeError(
                    f"full.md not found in zip. Files: {zip_file.namelist()}"
                )

            with zip_file.open(markdown_files[0]) as file:
                return file.read().decode("utf-8")
