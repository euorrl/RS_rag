import re
from pathlib import Path

from app.chunker import chunk_document
from app.embedder import embed_chunks
from app.reader import load_document
from app.vector_store import get_vector_store

DATA_DIR = Path("data/remote_sensing_fundamentals")
COLLECTION_NAME = "rag_chunks"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

EMBEDDING_PROVIDER = "bge"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIMENSION = 512


def page_start_key(path: Path) -> int:
    match = re.search(r"-(\d+)-\d+\.pdf$", path.name)
    if match:
        return int(match.group(1))
    return 10**9


def main() -> None:
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"), key=page_start_key)

    if not pdf_paths:
        print(f"No PDF files found in {DATA_DIR}")
        return

    vector_store = get_vector_store(
        provider="milvus",
        collection_name=COLLECTION_NAME,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        dimension=EMBEDDING_DIMENSION,
    )
    vector_store.drop_collection()
    print(f"Cleared Milvus collection: {COLLECTION_NAME}")

    all_embedded_chunks = []
    total_chunks = 0

    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")

        document = load_document(pdf_path)
        chunks = chunk_document(document)
        total_chunks += len(chunks)

        embedded_chunks = embed_chunks(
            chunks,
            provider=EMBEDDING_PROVIDER,
            model_name=EMBEDDING_MODEL,
            normalize_embeddings=True,
            use_instruction=False,
        )
        all_embedded_chunks.extend(embedded_chunks)

    if not all_embedded_chunks:
        print("No embedded chunks to write.")
        return

    if len(all_embedded_chunks[0].embedding) != EMBEDDING_DIMENSION:
        vector_store = get_vector_store(
            provider="milvus",
            collection_name=COLLECTION_NAME,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            dimension=len(all_embedded_chunks[0].embedding),
        )

    vector_store.insert(all_embedded_chunks)

    print("========== Index Summary ==========")
    print(f"PDF files: {len(pdf_paths)}")
    print(f"Chunks: {total_chunks}")
    print(f"Embedded chunks written: {len(all_embedded_chunks)}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
