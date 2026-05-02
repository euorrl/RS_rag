from pathlib import Path
from app.reader import TextReader


def main() -> None:
    reader = TextReader()
    doc = reader.read(Path("test.txt"))

    print("document_id:", doc.document_id)
    print("file_name:", doc.file_name)
    print("file_type:", doc.file_type)
    print("text:", doc.text)
    print("metadata:", doc.metadata)


if __name__ == "__main__":
    main()
