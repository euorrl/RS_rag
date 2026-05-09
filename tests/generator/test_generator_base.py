import pytest

from app.generator import BaseGenerator

pytestmark = pytest.mark.generator


def test_base_generator_cannot_be_instantiated():
    """验证 BaseGenerator 作为抽象基类不能被直接实例化。"""
    with pytest.raises(TypeError):
        BaseGenerator()


def test_subclass_without_generate_methods_cannot_be_instantiated():
    """验证未实现生成方法的子类不能被实例化。"""

    class MissingGenerateMethods(BaseGenerator):
        pass

    with pytest.raises(TypeError):
        MissingGenerateMethods()


def test_subclass_with_generate_methods_can_be_instantiated():
    """验证实现生成方法的子类可以正常实例化和调用。"""

    class SimpleGenerator(BaseGenerator):
        def generate(self, prompt):
            return "answer"

        def stream_generate(self, prompt):
            yield "answer"

    generator = SimpleGenerator()

    assert generator.generate("prompt") == "answer"
    assert list(generator.stream_generate("prompt")) == ["answer"]
