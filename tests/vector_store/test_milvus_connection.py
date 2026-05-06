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

    pymilvus.connections.connect(
        alias="default",
        host="localhost",
        port="19530",
    )

    assert pymilvus.utility.has_collection("test_collection") is False
