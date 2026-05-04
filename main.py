from app.reader import load_document
from app.chunker import chunk_document

doc = load_document("data/test_markdown.md")
chunks = chunk_document(doc)

for c in chunks[:]:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(c.text)
    print(c.metadata)
