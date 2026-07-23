import re


STANDARD_ITEM_TITLES = {
    "business": "1",
    "risk factors": "1A",
    "unresolved staff comments": "1B",
    "cybersecurity": "1C",
    "properties": "2",
    "legal proceedings": "3",
    "mine safety disclosures": "4",
    "market for registrant's common equity, related stockholder matters, and issuer purchases of equity securities": "5",
    "market for registrant's common equity, related stockholder matters and issuer purchases of equity securities": "5",
    "management's discussion and analysis of financial condition and results of operations": "7",
    "management's discussion and analysis": "7",
    "quantitative and qualitative disclosures about market risk": "7A",
    "financial statements and supplementary data": "8",
    "changes in and disagreements with accountants on accounting and financial disclosure": "9",
    "controls and procedures": "9A",
    "other information": "9B",
    "disclosure regarding foreign jurisdictions that prevent inspections": "9C",
    "directors, executive officers and corporate governance": "10",
    "directors, executive officers, and corporate governance": "10",
    "executive compensation": "11",
    "security ownership of certain beneficial owners and management and related stockholder matters": "12",
    "certain relationships and related transactions, and director independence": "13",
    "principal accountant fees and services": "14",
    "exhibit and financial statement schedules": "15",
    "exhibits and financial statement schedules": "15",
    "form 10-k summary": "16",
}


def normalize_title(text: str) -> str:
    t = text.lower().strip()
    t = t.rstrip(":.")
    t = re.sub(r"\s+", " ", t)
    return t


def classify_lines(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    item_pattern = re.compile(r"^item\s+\d+[a]?\.", re.IGNORECASE)

    classified = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        header_text = stripped.lstrip("#").strip()

        if stripped == "":
            line_type = "blank"
        elif item_pattern.match(header_text):
            line_type = "section_header"
        elif stripped.startswith("#") and normalize_title(header_text) in STANDARD_ITEM_TITLES:
            line_type = "section_header"
        elif stripped.startswith("|") and stripped.count("|") >= 2:
            line_type = "table_row"
        else:
            line_type = "prose"

        classified.append({
            "line_num": i,
            "text": stripped,
            "type": line_type
        })

    return classified


def group_into_blocks(classified_lines: list[dict]) -> list[dict]:
    separator_pattern = re.compile(r"[\|\-\s:]+")
    item_pattern = re.compile(r"item\s+(\d+[a]?)", re.IGNORECASE)

    blocks = []
    current_block_type = None
    current_block_lines = []
    current_start_line = None
    section_name = None
    item_number = None

    def close_block():
        nonlocal current_block_type, current_block_lines, current_start_line
        if current_block_type is not None and current_block_lines:
            blocks.append({
                "block_type": current_block_type,
                "section_name": section_name,
                "item_number": item_number,
                "lines": current_block_lines,
                "start_line": current_start_line
            })
        current_block_type = None
        current_block_lines = []
        current_start_line = None

    for entry in classified_lines:
        line_type = entry["type"]
        text = entry["text"]

        if line_type == "blank":
            if current_block_type == "table_row":
                close_block()
            continue

        if line_type == "section_header":
            close_block()
            header_text = text.lstrip("#").strip()
            section_name = header_text

            match = item_pattern.search(text)
            if match:
                item_number = match.group(1).upper()
            else:
                candidate = STANDARD_ITEM_TITLES.get(normalize_title(header_text))
                if candidate:
                    item_number = candidate
            continue

        if line_type == "table_row" and separator_pattern.fullmatch(text):
            continue

        if line_type != current_block_type:
            close_block()
            current_block_type = line_type
            current_start_line = entry["line_num"]

        current_block_lines.append(text)

    close_block()
    return blocks


if __name__ == "__main__":
    classified = classify_lines("extracted_text/apple_2024.md")
    blocks = group_into_blocks(classified)

    print("=== First 20 blocks (Apple) ===")
    for b in blocks[:20]:
        preview = b["lines"][0][:60] if b["lines"] else ""
        section = b["section_name"][:40] if b["section_name"] else "None"
        print(f"[{b['block_type']:<10}] item={b['item_number']} section={section!r:<42} lines={len(b['lines'])} :: {preview}")