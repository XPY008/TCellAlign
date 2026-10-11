from pathlib import Path


def read_prompt_section(path: Path, section: str) -> str:
    text = path.read_text(encoding="utf-8")
    _, separator, body = text.partition(f"# {section}\n\n")
    if not separator:
        raise ValueError(f"Missing prompt section {section!r}: {path}")
    return body.split("\n# ", 1)[0].rstrip()
