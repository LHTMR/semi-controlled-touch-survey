"""
Convert a Qualtrics .qsf survey file to a human-readable Markdown document.

Usage:
    python qsf_to_markdown.py <input.qsf> [output.md]

If output path is omitted, the Markdown is written next to the input file
with a .md extension.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path


# --- helpers -----------------------------------------------------------------

def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_placeholder(text: str) -> bool:
    """Return True for Qualtrics default placeholder strings."""
    markers = [
        "click to write",
        "click to edit",
        "&nbsp;",
    ]
    lower = text.lower()
    return any(m in lower for m in markers) or not text.strip()


QUESTION_TYPE_LABELS = {
    "MC": "Multiple choice",
    "TE": "Text entry",
    "Matrix": "Matrix",
    "HeatMap": "Heat map (image click)",
    "SBS": "Side-by-side matrix",
    "DB": "Display / instruction text",
    "Meta": "Hidden metadata",
    "Slider": "Slider",
    "Rank": "Rank order",
    "CS": "Constant sum",
    "RO": "Rank order",
}

SELECTOR_LABELS = {
    "SAVR": "single-answer",
    "MAVR": "multi-answer",
    "MSB": "multi-answer (checkbox)",
    "MACOL": "multi-answer (column)",
    "DL": "dropdown",
    "ESTB": "essay text box",
    "SL": "single line",
    "ML": "multi-line",
    "Likert": "Likert scale",
    "TB": None,  # suppress for display blocks
    "SBSMatrix": None,
}


def format_question_type(qt: str, sel: str) -> str:
    label = QUESTION_TYPE_LABELS.get(qt, qt)
    sel_label = SELECTOR_LABELS.get(sel)
    if sel_label:
        return f"{label} — {sel_label}"
    return label


def choices_to_list(choices: dict, order=None) -> list:
    if order:
        keys = [str(k) for k in order]
    else:
        keys = list(choices.keys())
    result = []
    for k in keys:
        v = choices.get(k, choices.get(int(k) if k.isdigit() else k, {}))
        if isinstance(v, dict):
            display = v.get("Display", "")
        else:
            display = str(v)
        display = strip_html(display)
        if display and not is_placeholder(display):
            result.append(display)
    return result


# --- block / question rendering ----------------------------------------------

def render_question(payload: dict, export_tag: str, index: int) -> list[str]:
    lines = []
    qt = payload.get("QuestionType", "")
    sel = payload.get("Selector", "")
    raw_text = payload.get("QuestionText", "")
    text = strip_html(raw_text)

    # Skip pure metadata / browser collectors
    if qt == "Meta":
        return []

    # Skip hidden flow-control questions (choices are all URLs)
    if qt == "MC":
        choices = payload.get("Choices", {})
        if choices:
            displays = [
                strip_html(v.get("Display", "") if isinstance(v, dict) else str(v))
                for v in choices.values()
            ]
            if all(d.startswith("http") for d in displays if d):
                return []

    # Skip display blocks that are empty placeholders
    if qt == "DB" and is_placeholder(text):
        return []

    # Section header
    type_label = format_question_type(qt, sel)
    tag_str = f"`{export_tag}`" if export_tag else ""
    header_parts = [f"**Q{index}**"]
    if tag_str:
        header_parts.append(tag_str)
    header_parts.append(f"*{type_label}*")
    lines.append("#### " + "  |  ".join(header_parts))
    lines.append("")

    if text and not is_placeholder(text):
        for para in text.split("\n"):
            para = para.strip()
            if para:
                lines.append(para)
                lines.append("")

    # Display / instruction — no choices to show
    if qt == "DB":
        return lines

    choices = payload.get("Choices", {})
    choice_order = payload.get("ChoiceOrder") or None
    answers = payload.get("Answers", {})
    answer_order = payload.get("AnswerOrder") or None

    if qt == "TE":
        lines.append("*Open text response*")
        lines.append("")

    elif qt in ("MC", "Rank", "CS", "RO"):
        items = choices_to_list(choices, choice_order)
        if items:
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    elif qt == "Matrix":
        row_items = choices_to_list(choices, choice_order)
        col_items = choices_to_list(answers, answer_order)
        if col_items:
            header = "| Statement | " + " | ".join(col_items) + " |"
            sep = "|---" * (len(col_items) + 1) + "|"
            lines.append(header)
            lines.append(sep)
            for row in row_items:
                lines.append("| " + row + " |" + " |" * len(col_items))
            lines.append("")

    elif qt == "HeatMap":
        lines.append("*Response: click location on image (valence × arousal grid)*")
        lines.append("")

    elif qt == "SBS":
        sub_qs = payload.get("AdditionalQuestions", {})
        for sq_key in sorted(sub_qs.keys(), key=lambda x: int(x) if x.isdigit() else x):
            sq = sub_qs[sq_key]
            col_text = strip_html(sq.get("QuestionText", ""))
            if col_text and not is_placeholder(col_text):
                lines.append(f"**Column: {col_text}**")
            sub_answers = sq.get("Answers", {})
            sub_choices = sq.get("Choices", {})
            col_items = choices_to_list(sub_answers)
            row_items = choices_to_list(sub_choices)
            if col_items and row_items:
                header = "| Statement | " + " | ".join(col_items) + " |"
                sep = "|---" * (len(col_items) + 1) + "|"
                lines.append(header)
                lines.append(sep)
                for row in row_items:
                    lines.append("| " + row + " |" + " |" * len(col_items))
            elif col_items:
                for item in col_items:
                    lines.append(f"- {item}")
            lines.append("")

    return lines


# --- main conversion ---------------------------------------------------------

def convert(qsf_path: Path) -> str:
    with open(qsf_path, encoding="utf-8") as f:
        data = json.load(f)

    entry = data.get("SurveyEntry", {})
    elements = data.get("SurveyElements", [])

    # Build QID → payload map
    sq_map = {
        e["PrimaryAttribute"]: e["Payload"]
        for e in elements
        if e["Element"] == "SQ"
    }

    # Get block structure
    bl = next((e for e in elements if e["Element"] == "BL"), None)
    if bl is None:
        sys.exit("No block element (BL) found in .qsf file.")
    blocks = bl["Payload"]

    # Sort blocks by their integer key, skipping Trash
    def block_sort_key(item):
        k, v = item
        try:
            return int(k)
        except ValueError:
            return 9999

    ordered_blocks = [
        (k, v) for k, v in sorted(blocks.items(), key=block_sort_key)
        if isinstance(v, dict) and v.get("Type") != "Trash"
    ]

    # Markdown output
    out = []
    survey_name = entry.get("SurveyName", qsf_path.stem)
    out.append(f"# {survey_name}")
    out.append("")
    out.append(
        f"Exported from Qualtrics survey file `{qsf_path.name}` "
        f"(survey ID: `{entry.get('SurveyID', 'unknown')}`)."
    )
    out.append("")
    out.append(
        "Questions marked *Display / instruction text* are shown to participants "
        "as read-only text. Questions marked *Hidden metadata* are not shown to "
        "participants and are omitted here."
    )
    out.append("")

    q_index = 0

    for _k, block in ordered_blocks:
        block_desc = block.get("Description", "")
        block_elements = block.get("BlockElements", [])
        question_ids = [
            be["QuestionID"]
            for be in block_elements
            if be["Type"] == "Question"
        ]
        if not question_ids:
            continue

        # Collapse repeated video blocks into one representative section
        is_video_block = re.match(
            r"^(Video|\d+(st|nd|rd|th) video)$", block_desc, re.IGNORECASE
        )

        out.append(f"---")
        out.append("")
        out.append(f"## {block_desc}")
        out.append("")

        if is_video_block and block_desc.lower() != "video":
            out.append(
                "_This block is repeated for each video stimulus. "
                "Only the first instance (block 'Video') is shown in full above._"
            )
            out.append("")
            continue

        for qid in question_ids:
            payload = sq_map.get(qid)
            if payload is None:
                continue
            export_tag = payload.get("DataExportTag", "")
            q_index += 1
            rendered = render_question(payload, export_tag, q_index)
            if not rendered:
                q_index -= 1  # don't count skipped questions
                continue
            out.extend(rendered)

    return "\n".join(out)


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Qualtrics .qsf file to readable Markdown."
    )
    parser.add_argument("input", help="Path to the .qsf file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .md path (default: same directory as input, .md extension)",
    )
    args = parser.parse_args()

    qsf_path = Path(args.input).resolve()
    if not qsf_path.exists():
        sys.exit(f"File not found: {qsf_path}")

    out_path = Path(args.output).resolve() if args.output else qsf_path.with_suffix(".md")

    md = convert(qsf_path)
    out_path.write_text(md, encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
