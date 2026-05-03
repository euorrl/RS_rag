from app.reader import load_document


def test_file(file_path: str):
    """测试单个文件"""
    print("\n" + "=" * 50)
    print(f"Testing: {file_path}")

    try:
        doc = load_document(file_path)

        print("File:", doc.file_name)
        print("Type:", doc.file_type)
        print("Metadata:", doc.metadata)
        print("text:")
        print(doc.text)

    except Exception as e:
        print("❌ Error:", str(e))


def main():
    """测试所有文件类型"""

    files = [
        "data/test_txt.txt",  # txt
        "data/test_markdown.md",  # markdown
        "data/test.png",  # image (OCR)
        "data/遥感导论-目录-11-16.pdf",  # pdf
    ]

    for file in files:
        test_file(file)


if __name__ == "__main__":
    main()
