from app.reader import load_document
from app.chunker import chunk_document

doc = load_document("data/遥感导论-第1章-17-29.pdf")
chunks = chunk_document(doc)

for c in chunks[:]:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(c.text)
    print(c.metadata)
