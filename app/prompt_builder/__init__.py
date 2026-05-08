from typing import TYPE_CHECKING

from app.prompt_builder.prompt_builder_base import BasePromptBuilder

if TYPE_CHECKING:  # pragma: no cover
    from app.prompt_builder.chat_messages_prompt_builder import (
        ChatMessagesPromptBuilder as ChatMessagesPromptBuilder,
    )
    from app.prompt_builder.prompt_builder_factory import (
        build_messages_prompt as build_messages_prompt,
    )
    from app.prompt_builder.prompt_builder_factory import (
        build_prompt as build_prompt,
    )
    from app.prompt_builder.prompt_builder_factory import (
        build_string_prompt as build_string_prompt,
    )
    from app.prompt_builder.prompt_builder_factory import (
        get_prompt_builder as get_prompt_builder,
    )
    from app.prompt_builder.string_prompt_builder import (
        StringPromptBuilder as StringPromptBuilder,
    )

__all__ = [
    "BasePromptBuilder",
    "ChatMessagesPromptBuilder",
    "StringPromptBuilder",
    "get_prompt_builder",
    "build_prompt",
    "build_messages_prompt",
    "build_string_prompt",
]


def __getattr__(name: str):
    if name == "ChatMessagesPromptBuilder":
        from app.prompt_builder.chat_messages_prompt_builder import (
            ChatMessagesPromptBuilder,
        )

        return ChatMessagesPromptBuilder

    if name == "StringPromptBuilder":
        from app.prompt_builder.string_prompt_builder import StringPromptBuilder

        return StringPromptBuilder

    if name in {"get_prompt_builder", "build_prompt"}:
        from app.prompt_builder.prompt_builder_factory import (
            build_prompt,
            get_prompt_builder,
        )

        return {
            "get_prompt_builder": get_prompt_builder,
            "build_prompt": build_prompt,
        }[name]

    if name in {"build_messages_prompt", "build_string_prompt"}:
        from app.prompt_builder.prompt_builder_factory import (
            build_messages_prompt,
            build_string_prompt,
        )

        return {
            "build_messages_prompt": build_messages_prompt,
            "build_string_prompt": build_string_prompt,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
