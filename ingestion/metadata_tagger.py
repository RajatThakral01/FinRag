import os
import re
from line_classifier import classify_lines, group_into_blocks
from chunker import chunk_prose_block, chunk_table_block, count_tokens
from parent_linker import link_parent_chunks

COMPANY_MAP = {
    "apple":     ("Apple Inc.", "AAPL"),
    "microsoft": ("Microsoft Corporation", "MSFT"),
    "amazon":    ("Amazon.com Inc.", "AMZN"),
    "nvidia":    ("NVIDIA Corporation", "NVDA"),
    "tesla":     ("Tesla Inc.", "TSLA"),
    "meta":      ("Meta Platforms Inc.", "META"),
    "alphabet":  ("Alphabet Inc.", "GOOGL"),
    "netflix":   ("Netflix Inc.", "NFLX"),
    "adobe":     ("Adobe Inc.", "ADBE"),
}


def get_company_info(filepath: str) -> tuple[str, str, str]:
    basename = os.path.basename(filepath)
    slug = basename.split("_")[0].lower()

    if slug not in COMPANY_MAP:
        raise ValueError(f"Unrecognized company slug '{slug}' from filename '{basename}'")

    company, ticker = COMPANY_MAP[slug]

    year_match = re.search(r"(20\d{2})", basename)
    year = year_match.group(1) if year_match else "unknown"

    return company, ticker, year


def extract_table_name(blocks: list[dict], table_block_index: int) -> str:
    if table_block_index > 0:
        prev_block = blocks[table_block_index - 1]
        if prev_block["block_type"] == "prose" and prev_block["lines"]:
            for line in reversed(prev_block["lines"]):
                if line.startswith("#"):
                    return line.lstrip("#").strip()

    return blocks[table_block_index]["section_name"] or "Unnamed Table"


def extract_columns(header_row: str) -> str:
    cells = [c.strip() for c in header_row.split("|")]
    cells = [c for c in cells if c]
    return ", ".join(cells)


def tag_all_chunks(filepath: str) -> list[dict]:
    company, ticker, year = get_company_info(filepath)

    classified = classify_lines(filepath)
    blocks = group_into_blocks(classified)

    tagged_chunks = []

    for block_idx, block in enumerate(blocks):
        base_meta = {
            "company": company,
            "ticker": ticker,
            "year": year,
            "item_number": block["item_number"],
            "section_name": block["section_name"],
            "page_start": None,
            "block_idx": block_idx,
            "parent_chunk_id": None,
        }

        if block["block_type"] == "prose":
            texts = chunk_prose_block(block)
            for chunk_idx, text in enumerate(texts):
                chunk_id = f"{ticker.lower()}_{year}_item{block['item_number'] or 'na'}_prose_{block_idx:03d}_{chunk_idx:03d}"
                tagged_chunks.append({
                    **base_meta,
                    "chunk_type": "prose",
                    "table_name": None,
                    "chunk_id": chunk_id,
                    "text": text,
                })

        elif block["block_type"] == "table_row":
            if not block["lines"]:
                continue

            table_name = extract_table_name(blocks, block_idx)
            columns = extract_columns(block["lines"][0])
            header_line = f"{company} | {year} | {table_name} | Columns: {columns}"

            texts = chunk_table_block(block, header_line)
            for chunk_idx, text in enumerate(texts):
                chunk_id = f"{ticker.lower()}_{year}_item{block['item_number'] or 'na'}_table_{block_idx:03d}_{chunk_idx:03d}"
                tagged_chunks.append({
                    **base_meta,
                    "chunk_type": "table",
                    "table_name": table_name,
                    "chunk_id": chunk_id,
                    "text": text,
                })

    tagged_chunks = link_parent_chunks(tagged_chunks)
    return tagged_chunks


def tag_all_companies(folder: str) -> dict[str, list[dict]]:
    results = {}

    for slug, (company, ticker) in COMPANY_MAP.items():
        filename = f"{slug}_2024.md"
        filepath = os.path.join(folder, filename)

        try:
            chunks = tag_all_chunks(filepath)
            results[ticker] = chunks
        except Exception as e:
            print(f"FAILED: {company} ({ticker}) — file '{filename}' — {type(e).__name__}: {e}")
            continue

    return results


if __name__ == "__main__":
    all_results = tag_all_companies("extracted_text")

    print(f"\n{len(all_results)} of {len(COMPANY_MAP)} companies processed successfully\n")

    for ticker, chunks in all_results.items():
        linked = sum(1 for c in chunks if c["parent_chunk_id"] is not None)
        print(f"{ticker}: total={len(chunks)} chunks_with_parent={linked}")

    print("\n=== Sample linked chunks (Apple) ===")
    aapl_linked = [c for c in all_results["AAPL"] if c["parent_chunk_id"] is not None]
    for c in aapl_linked[:5]:
        print(f"chunk_id={c['chunk_id']} (type={c['chunk_type']}) -> parent={c['parent_chunk_id']}")           
                