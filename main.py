from app.reader.markdown_reader import MarkdownReader


def test_markdown_reader():
    """手动验证 MarkdownReader 读取复杂 Markdown 文件。"""
    reader = MarkdownReader()
    doc = reader.read("data/test_markdown.md")

    print("=== Markdown Reader ===")
    print(doc.file_name)
    print(doc.file_type)
    print(doc.metadata)
    print()
    print(doc.text[:])


if __name__ == "__main__":
    test_markdown_reader()
