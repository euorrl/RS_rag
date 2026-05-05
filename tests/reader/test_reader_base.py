import pytest

from app.reader import BaseReader

pytestmark = pytest.mark.reader


def test_base_reader_cannot_be_instantiated():
    """验证 BaseReader 作为抽象类无法被实例化。"""
    with pytest.raises(TypeError):
        BaseReader()


def test_subclass_without_read_cannot_be_instantiated():
    """验证未实现 read 方法的子类无法实例化。"""

    class BadReader(BaseReader):
        pass

    with pytest.raises(TypeError):
        BadReader()


def test_subclass_with_read_can_be_instantiated():
    """验证实现了 read 方法的子类可以正常实例化。"""

    class GoodReader(BaseReader):
        def read(self, file_path):
            return None

    reader = GoodReader()
    assert isinstance(reader, BaseReader)
