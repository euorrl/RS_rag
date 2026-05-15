import sys
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import utility

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vector_store.milvus_store import MilvusVectorStore  # noqa: E402


def main() -> None:
    """检查当前 Milvus 连接配置，且不打印任何密钥。

    该脚本会读取 ``.env``，初始化 ``MilvusVectorStore``，打印当前 URI、
    数据库名和 collection 名称，并列出现有 collections。如果目标 collection
    不存在，只打印简短提示。
    """
    load_dotenv()

    try:
        vector_store = MilvusVectorStore()
        collections = utility.list_collections()
    except Exception as exc:
        print(f"Milvus 连接失败：{exc.__class__.__name__}")
        return

    print(f"MILVUS_MODE: {vector_store.mode}")
    print(f"MILVUS_URI: {vector_store.uri}")
    print(f"MILVUS_DB_NAME: {vector_store.db_name}")
    print(f"MILVUS_COLLECTION_NAME: {vector_store.collection_name}")
    print(f"已有 collections：{collections}")

    if vector_store.collection_name not in collections:
        print(f"未找到 collection：{vector_store.collection_name}")


if __name__ == "__main__":
    main()
