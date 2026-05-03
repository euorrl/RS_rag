import io
import zipfile

import pytest

from app.reader.mineru_client import MinerUClient

pytestmark = pytest.mark.reader


class MockResponse:
    """模拟 requests 返回对象。"""

    def __init__(self, json_data=None, content=b"", status_code=200, text=""):
        """初始化模拟响应。

        Args:
            json_data (dict | None): 模拟 JSON 响应数据。
            content (bytes): 模拟二进制响应内容。
            status_code (int): HTTP 状态码。
            text (str): 响应文本。
        """
        self._json_data = json_data
        self.content = content
        self.status_code = status_code
        self.text = text

    def json(self):
        """返回模拟 JSON 数据。

        Returns:
            dict: 模拟 JSON 响应。
        """
        return self._json_data

    def raise_for_status(self):
        """模拟 requests 的 raise_for_status 方法。

        Returns:
            None
        """
        return None


def create_test_zip(markdown: str) -> bytes:
    """创建包含 full.md 的测试 ZIP。

    Args:
        markdown (str): 写入 full.md 的 Markdown 文本。

    Returns:
        bytes: ZIP 文件的二进制内容。
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, mode="w") as zip_file:
        zip_file.writestr("output/full.md", markdown)

    return buffer.getvalue()


def test_mineru_client_parse_file_success(monkeypatch, tmp_path):
    """验证 MinerUClient 能完成完整解析流程。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content")

    markdown = "# 遥感导论\n\nNDVI = test"
    zip_content = create_test_zip(markdown)

    def mock_post(url, headers=None, json=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 0,
                "data": {
                    "batch_id": "batch-123",
                    "file_urls": ["https://upload.example.com/test.pdf"],
                },
            }
        )

    def mock_put(url, data=None, timeout=None):
        return MockResponse(status_code=200)

    def mock_get(url, headers=None, timeout=None):
        if "extract-results" in url:
            download_url = "https://download.example.com/result.zip"
            return MockResponse(
                json_data={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "state": "done",
                                "full_zip_url": download_url,
                            }
                        ]
                    },
                }
            )

        return MockResponse(content=zip_content)

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.put", mock_put)
    monkeypatch.setattr("requests.get", mock_get)

    client = MinerUClient()
    result = client.parse_file(file_path)

    assert result == markdown


def test_mineru_client_raises_error_without_token(monkeypatch):
    """验证缺少 MINERU_API_TOKEN 时抛出 RuntimeError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "")

    with pytest.raises(RuntimeError, match="Missing MINERU_API_TOKEN"):
        MinerUClient()


def test_mineru_client_raises_error_when_apply_upload_url_failed(monkeypatch, tmp_path):
    """验证申请上传 URL 失败时抛出 RuntimeError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content")

    def mock_post(url, headers=None, json=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 1001,
                "msg": "invalid request",
            }
        )

    monkeypatch.setattr("requests.post", mock_post)

    client = MinerUClient()

    with pytest.raises(RuntimeError, match="MinerU apply upload url failed"):
        client.parse_file(file_path)


def test_mineru_client_raises_error_when_upload_failed(monkeypatch, tmp_path):
    """验证文件上传失败时抛出 RuntimeError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content")

    def mock_post(url, headers=None, json=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 0,
                "data": {
                    "batch_id": "batch-123",
                    "file_urls": ["https://upload.example.com/test.pdf"],
                },
            }
        )

    def mock_put(url, data=None, timeout=None):
        return MockResponse(status_code=500, text="upload error")

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.put", mock_put)

    client = MinerUClient()

    with pytest.raises(RuntimeError, match="MinerU file upload failed"):
        client.parse_file(file_path)


def test_mineru_client_raises_error_when_parse_failed(monkeypatch, tmp_path):
    """验证 MinerU 解析失败时抛出 RuntimeError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。
        tmp_path (pathlib.Path): pytest 提供的临时目录。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake pdf content")

    def mock_post(url, headers=None, json=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 0,
                "data": {
                    "batch_id": "batch-123",
                    "file_urls": ["https://upload.example.com/test.pdf"],
                },
            }
        )

    def mock_put(url, data=None, timeout=None):
        return MockResponse(status_code=200)

    def mock_get(url, headers=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 0,
                "data": {
                    "extract_result": [
                        {
                            "state": "failed",
                            "err_msg": "parse error",
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.put", mock_put)
    monkeypatch.setattr("requests.get", mock_get)

    client = MinerUClient()

    with pytest.raises(RuntimeError, match="MinerU parse failed"):
        client.parse_file(file_path)


def test_mineru_client_raises_error_when_full_md_not_found(monkeypatch):
    """验证 ZIP 中不存在 full.md 时抛出 RuntimeError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zip_file:
        zip_file.writestr("output/other.md", "# other")

    def mock_get(url, timeout=None):
        return MockResponse(content=buffer.getvalue())

    monkeypatch.setattr(
        "app.reader.mineru_client.requests.get",
        mock_get,
    )

    client = MinerUClient()

    with pytest.raises(RuntimeError, match="full.md not found"):
        client._download_markdown("https://download.example.com/result.zip")


def test_mineru_client_raises_error_when_query_failed(monkeypatch):
    """验证查询解析结果失败时抛出 RuntimeError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    def mock_get(url, headers=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 1001,
                "msg": "query error",
            }
        )

    monkeypatch.setattr("app.reader.mineru_client.requests.get", mock_get)

    client = MinerUClient()

    with pytest.raises(RuntimeError, match="MinerU query failed"):
        client._poll_result("batch-123")


def test_mineru_client_poll_result_waits_when_result_is_empty(monkeypatch):
    """验证解析结果为空时会继续轮询，直到解析完成。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    responses = [
        {
            "code": 0,
            "data": {
                "extract_result": [],
            },
        },
        {
            "code": 0,
            "data": {
                "extract_result": [
                    {
                        "state": "done",
                        "full_zip_url": "https://download.example.com/result.zip",
                    }
                ]
            },
        },
    ]

    def mock_get(url, headers=None, timeout=None):
        return MockResponse(json_data=responses.pop(0))

    monkeypatch.setattr("app.reader.mineru_client.requests.get", mock_get)
    monkeypatch.setattr("app.reader.mineru_client.time.sleep", lambda seconds: None)

    client = MinerUClient()
    result = client._poll_result("batch-123")

    assert result == "https://download.example.com/result.zip"


def test_mineru_client_poll_result_waits_when_state_is_running(monkeypatch):
    """验证任务处于 running 状态时会继续轮询。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    responses = [
        {
            "code": 0,
            "data": {
                "extract_result": [
                    {
                        "state": "running",
                    }
                ]
            },
        },
        {
            "code": 0,
            "data": {
                "extract_result": [
                    {
                        "state": "done",
                        "full_zip_url": "https://download.example.com/result.zip",
                    }
                ]
            },
        },
    ]

    def mock_get(url, headers=None, timeout=None):
        return MockResponse(json_data=responses.pop(0))

    monkeypatch.setattr("app.reader.mineru_client.requests.get", mock_get)
    monkeypatch.setattr("app.reader.mineru_client.time.sleep", lambda seconds: None)

    client = MinerUClient()
    result = client._poll_result("batch-123")

    assert result == "https://download.example.com/result.zip"


def test_mineru_client_raises_timeout_when_parse_timeout(monkeypatch):
    """验证轮询超时时抛出 TimeoutError。

    Args:
        monkeypatch (pytest.MonkeyPatch): pytest 提供的猴子补丁工具。

    Returns:
        None
    """
    monkeypatch.setenv("MINERU_API_TOKEN", "fake-token")

    def mock_get(url, headers=None, timeout=None):
        return MockResponse(
            json_data={
                "code": 0,
                "data": {
                    "extract_result": [
                        {
                            "state": "running",
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr("app.reader.mineru_client.requests.get", mock_get)
    monkeypatch.setattr("app.reader.mineru_client.time.sleep", lambda seconds: None)

    client = MinerUClient()

    with pytest.raises(TimeoutError, match="MinerU parse timeout"):
        client._poll_result("batch-123")
