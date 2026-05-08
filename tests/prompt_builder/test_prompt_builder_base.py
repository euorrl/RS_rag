import pytest

from app.prompt_builder import BasePromptBuilder

pytestmark = pytest.mark.prompt_builder


def test_base_prompt_builder_cannot_be_instantiated():
    """验证 BasePromptBuilder 作为抽象基类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BasePromptBuilder()


def test_subclass_without_build_cannot_be_instantiated():
    """验证未实现 build 方法的子类不能被实例化。"""

    class MissingBuild(BasePromptBuilder):
        pass

    with pytest.raises(TypeError):
        MissingBuild()


def test_subclass_with_build_can_be_instantiated():
    """验证实现 build 方法的子类可以正常实例化和调用。"""

    class SimplePromptBuilder(BasePromptBuilder):
        def build(self, query: str, retrieved_chunks: list, history=None):
            return query

    prompt_builder = SimplePromptBuilder()

    assert prompt_builder.build("query", []) == "query"
