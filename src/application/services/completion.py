from typing import Protocol


class TokenizerVocabulary(Protocol):
    def get_vocab(self) -> dict[str, int]: ...


def build_fim_prompt(
    tokenizer: TokenizerVocabulary,
    prefix: str,
    suffix: str | None,
) -> str:
    """Build a fill-in-the-middle prompt for a tokenizer that supports it."""
    if suffix is None:
        return prefix

    vocabulary = tokenizer.get_vocab()
    token_sets = (
        ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
        ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    )
    for prefix_token, suffix_token, middle_token in token_sets:
        if all(
            token in vocabulary for token in (prefix_token, suffix_token, middle_token)
        ):
            return f"{prefix_token}{prefix}{suffix_token}{suffix}{middle_token}"
    return prefix


def truncate_at_stop(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    if stop is None:
        return text, False
    stops = [stop] if isinstance(stop, str) else stop
    positions = [
        position for value in stops if value and (position := text.find(value)) >= 0
    ]
    if not positions:
        return text, False
    return text[: min(positions)], True
