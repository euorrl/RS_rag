from app.reader.mineru_image_reader import MinerUImageReader


def test_image_reader():
    reader = MinerUImageReader()
    doc = reader.read("data/test.png")

    print("=== Image Reader ===")
    print(doc.file_name)
    print(doc.file_type)
    print(doc.metadata)
    print(doc.text[:1000])


if __name__ == "__main__":
    test_image_reader()
