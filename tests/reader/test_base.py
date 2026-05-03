import pytest

from app.reader import BaseReader

pytestmark = pytest.mark.reader


def test_base_reader_read_raises_not_implemented_error():
    """验证 BaseReader.read() 会抛出 NotImplementedError。

    Returns:
        None
    """
    reader = BaseReader()

    with pytest.raises(NotImplementedError):
        reader.read("test.txt")
