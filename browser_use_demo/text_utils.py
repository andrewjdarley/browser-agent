"""Small text helpers shared between the Streamlit renderer and the run logger."""


def clean_text_extraction_markers(text: str) -> str:
    """Remove text extraction markers and return a summary.

    read_page/get_page_text results are wrapped as
    "__PAGE_EXTRACTED__\\n{summary}\\n__FULL_CONTENT__\\n{full_content}" so the
    full content reaches the model but only the summary needs to be shown to
    a human (in the UI or in a log).
    """
    if "__PAGE_EXTRACTED__" not in text and "__TEXT_EXTRACTED__" not in text:
        return text

    lines = text.split("\n")
    summary = []
    for line in lines:
        if "__FULL_CONTENT__" in line:
            break
        if "__PAGE_EXTRACTED__" not in line and "__TEXT_EXTRACTED__" not in line:
            summary.append(line)
    return "\n".join(summary) + "\n[Full content extracted but truncated for readability]"
