import pytest

from app.reader import BaseReader

pytestmark = pytest.mark.reader


def test_base_reader_cannot_be_instantiated():
    """验证 BaseReader 作为抽象类无法被实例化。"""
    with pytest.raises(TypeError):
        BaseReader()
