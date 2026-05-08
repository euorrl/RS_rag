import pytest

from app.prompt_builder.string_prompt_builder import StringPromptBuilder
from app.schemas import RetrievedChunk

pytestmark = pytest.mark.prompt_builder


def make_retrieved_chunks() -> list[RetrievedChunk]:
    """创建测试用 RetrievedChunk 列表。"""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="NDVI 是归一化植被指数。",
            score=0.95,
            metadata={
                "file_name": "remote_sensing.md",
                "chunk_index": 1,
                "headers": {
                    "h1": "遥感指数",
                    "h2": "植被指数",
                    "h3": None,
                },
                "header_path": "遥感指数 > 植被指数",
            },
            recall_method="vector",
            rerank_method="bge",
            score_details={
                "vector_score": 0.8,
                "rerank_score": 0.95,
            },
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="NDVI 常用于植被长势监测。",
            score=0.9,
            metadata={
                "source_path": "data/rs.md",
                "chunk_index": 2,
                "headers": {
                    "h1": "遥感应用",
                    "h2": None,
                    "h3": None,
                },
            },
        ),
    ]


def test_string_prompt_builder_builds_prompt_with_contexts():
    """验证 StringPromptBuilder 会将 query 和候选 chunks 组织为字符串 prompt。"""
    prompt_builder = StringPromptBuilder()

    prompt = prompt_builder.build("什么是 NDVI?", make_retrieved_chunks())

    assert "你是一个严谨的知识问答助手" in prompt
    assert "参考资料：" in prompt
    assert "用户问题：" in prompt
    assert "什么是 NDVI?" in prompt
    assert "[资料 1]" in prompt
    assert "[资料 2]" in prompt
    assert "remote_sensing.md" in prompt
    assert "data/rs.md" in prompt
    assert "标题: 遥感指数 > 植被指数" in prompt
    assert "标题: 遥感应用" in prompt
    assert "文本块ID" not in prompt
    assert "文档ID" not in prompt
    assert "score_details" not in prompt
    assert "embedding" not in prompt
    assert "NDVI 是归一化植被指数。" in prompt
    assert "请基于以上要求作答。" in prompt
    assert "来源：资料文件名，标题：标题路径" in prompt


def test_string_prompt_builder_uses_header_path_first():
    """验证标题会优先使用 metadata 中已经拼好的 header_path。"""
    prompt_builder = StringPromptBuilder()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="text",
            score=0.9,
            metadata={
                "headers": {
                    "h1": "一级标题",
                    "h2": "二级标题",
                    "h3": None,
                },
                "header_path": "已经拼好的标题路径",
            },
        )
    ]

    prompt = prompt_builder.build("query", chunks)

    assert "标题: 已经拼好的标题路径" in prompt
    assert "标题: 一级标题 > 二级标题" not in prompt


def test_string_prompt_builder_uses_nested_headers_when_no_header_path():
    """验证没有 header_path 时会从 metadata.headers 中拼接标题层级。"""
    prompt_builder = StringPromptBuilder()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="text",
            score=0.9,
            metadata={
                "headers": {
                    "h1": "一级标题",
                    "h2": "二级标题",
                    "h3": "三级标题",
                },
            },
        )
    ]

    prompt = prompt_builder.build("query", chunks)

    assert "标题: 一级标题 > 二级标题 > 三级标题" in prompt


def test_string_prompt_builder_keeps_flat_header_fields_compatible():
    """验证旧版平铺 h1/h2/h3 标题字段仍然可以被识别。"""
    prompt_builder = StringPromptBuilder()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="text",
            score=0.9,
            metadata={
                "h1": "旧版一级标题",
                "h2": "旧版二级标题",
                "h3": None,
            },
        )
    ]

    prompt = prompt_builder.build("query", chunks)

    assert "标题: 旧版一级标题 > 旧版二级标题" in prompt


def test_string_prompt_builder_uses_custom_system_prompt():
    """验证 StringPromptBuilder 支持传入自定义系统提示词。"""
    prompt_builder = StringPromptBuilder(system_prompt="请严谨回答。")

    prompt = prompt_builder.build("query", [])

    assert prompt.startswith("请严谨回答。")
    assert "无可用参考资料。" in prompt


def test_string_prompt_builder_allows_general_fallback_by_default():
    """验证默认提示词允许在资料不足时使用通用知识补充。"""
    prompt_builder = StringPromptBuilder()

    prompt = prompt_builder.build("query", [])

    assert prompt_builder.allow_general_fallback is True
    assert "可以基于你的通用知识进行补充说明" in prompt
    assert "哪些内容是通用知识补充" in prompt
    assert "来源：资料文件名，标题：标题路径" in prompt


def test_string_prompt_builder_can_disable_general_fallback():
    """验证 allow_general_fallback 为 False 时会切换为严格依据资料模式。"""
    prompt_builder = StringPromptBuilder(allow_general_fallback=False)

    prompt = prompt_builder.build("query", [])

    assert prompt_builder.allow_general_fallback is False
    assert "请严格依据参考资料回答用户问题" in prompt
    assert "不要使用参考资料之外的通用知识进行补充或推断" in prompt
    assert "可以基于你的通用知识进行补充说明" not in prompt
    assert "来源：资料文件名，标题：标题路径" in prompt


def test_string_prompt_builder_returns_no_context_prompt_for_empty_chunks():
    """验证候选 chunks 为空时仍会生成可用字符串 prompt。"""
    prompt_builder = StringPromptBuilder()

    prompt = prompt_builder.build("什么是 NDVI?", [])

    assert "无可用参考资料。" in prompt
    assert "什么是 NDVI?" in prompt


def test_string_prompt_builder_truncates_long_context():
    """验证参考资料超过 max_context_chars 时会被截断。"""
    prompt_builder = StringPromptBuilder(max_context_chars=20)

    prompt = prompt_builder.build("什么是 NDVI?", make_retrieved_chunks())

    assert "...[参考资料已截断]" in prompt


def test_string_prompt_builder_can_disable_context_truncation():
    """验证 max_context_chars 为 None 时不会截断参考资料。"""
    prompt_builder = StringPromptBuilder(max_context_chars=None)

    prompt = prompt_builder.build("什么是 NDVI?", make_retrieved_chunks())

    assert "...[参考资料已截断]" not in prompt
    assert "NDVI 常用于植被长势监测。" in prompt


def test_string_prompt_builder_uses_unknown_source_when_metadata_missing():
    """验证 metadata 中没有来源和标题信息时会使用 unknown 占位。"""
    prompt_builder = StringPromptBuilder()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id=None,
            text="text",
            score=0.5,
        )
    ]

    prompt = prompt_builder.build("query", chunks)

    assert "来源: unknown" in prompt
    assert "标题: unknown" in prompt


def test_string_prompt_builder_rejects_invalid_init_input():
    """验证 StringPromptBuilder 会拒绝非法初始化参数。"""
    with pytest.raises(TypeError):
        StringPromptBuilder(system_prompt=123)

    with pytest.raises(ValueError):
        StringPromptBuilder(system_prompt="   ")

    with pytest.raises(TypeError):
        StringPromptBuilder(max_context_chars="100")

    with pytest.raises(ValueError):
        StringPromptBuilder(max_context_chars=0)

    with pytest.raises(TypeError):
        StringPromptBuilder(allow_general_fallback="true")


def test_string_prompt_builder_rejects_invalid_build_input():
    """验证 StringPromptBuilder 会拒绝非法 query、retrieved_chunks 和 history。"""
    prompt_builder = StringPromptBuilder()

    with pytest.raises(TypeError):
        prompt_builder.build(123, [])

    with pytest.raises(ValueError):
        prompt_builder.build("   ", [])

    with pytest.raises(TypeError):
        prompt_builder.build("query", "not-list")

    with pytest.raises(TypeError):
        prompt_builder.build("query", ["not-retrieved-chunk"])

    with pytest.raises(TypeError):
        prompt_builder.build("query", [], history=[])
