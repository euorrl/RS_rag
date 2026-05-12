import sys
from pathlib import Path
from app.vector_store.milvus_store import MilvusVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CHUNK_ID = "30b00ada-5ca8-4622-b998-2edd1d70457d"


def main() -> None:
    """查询指定 chunk_id 对应的原文并打印。"""
    vector_store = MilvusVectorStore()
    text = vector_store.get_chunk_text_by_id(CHUNK_ID)
    print(text)


if __name__ == "__main__":
    main()
