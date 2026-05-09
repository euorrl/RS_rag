from typing import TYPE_CHECKING

from app.generator.generator_base import BaseGenerator

if TYPE_CHECKING:  # pragma: no cover
    from app.generator.generator_factory import generate as generate
    from app.generator.generator_factory import get_generator as get_generator
    from app.generator.generator_factory import stream_generate as stream_generate
    from app.generator.openai_generator import OpenAIGenerator as OpenAIGenerator

__all__ = [
    "BaseGenerator",
    "OpenAIGenerator",
    "get_generator",
    "generate",
    "stream_generate",
]


def __getattr__(name: str):
    if name == "OpenAIGenerator":
        from app.generator.openai_generator import OpenAIGenerator

        return OpenAIGenerator

    if name in {"get_generator", "generate", "stream_generate"}:
        from app.generator.generator_factory import (
            generate,
            get_generator,
            stream_generate,
        )

        return {
            "get_generator": get_generator,
            "generate": generate,
            "stream_generate": stream_generate,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
