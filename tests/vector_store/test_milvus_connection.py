import os

import pytest
from dotenv import load_dotenv

pytestmark = [
    pytest.mark.vector_store,
    pytest.mark.integration,
]


def test_connect_milvus():
    load_dotenv()

    if os.getenv("RUN_MILVUS_TESTS") != "1":
        pytest.skip("Set RUN_MILVUS_TESTS=1 to run Milvus integration tests.")

    pymilvus = pytest.importorskip("pymilvus")

    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    token = os.getenv("MILVUS_TOKEN", "")
    db_name = os.getenv("MILVUS_DB_NAME", "default")

    connect_kwargs = {
        "alias": "default",
        "uri": uri,
        "db_name": db_name,
    }
    if token.strip():
        connect_kwargs["token"] = token

    try:
        pymilvus.connections.connect(**connect_kwargs)
    except Exception as exc:
        pytest.skip(f"Milvus is unavailable: {exc.__class__.__name__}")

    assert pymilvus.utility.has_collection("test_collection") is False
