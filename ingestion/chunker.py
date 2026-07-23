import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from line_classifier import classify_lines, group_into_blocks

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def chunk_prose_block(block: dict) -> list[str]:
    joined_text = "\n\n".join(block["lines"])

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=450,
        chunk_overlap=50
    )

    return splitter.split_text(joined_text)


def chunk_table_block(block: dict, header_line: str) -> list[str]:
    chunks = []
    current_lines = [header_line]

    for row in block["lines"]:
        candidate_lines = current_lines + [row]
        candidate_text = "\n".join(candidate_lines)

        if count_tokens(candidate_text) > 450 and len(current_lines) > 1:
            chunks.append("\n".join(current_lines))
            current_lines = [header_line, row]
        else:
            current_lines = candidate_lines

    if len(current_lines) > 1:
        chunks.append("\n".join(current_lines))

    return chunks


if __name__ == "__main__":
    classified = classify_lines("extracted_text/apple_2024.md")
    blocks = group_into_blocks(classified)

    print("=== Prose chunking: block index 4 (Item 1A) ===")
    prose_chunks = chunk_prose_block(blocks[4])
    print(f"Produced {len(prose_chunks)} chunks")
    for i, c in enumerate(prose_chunks):
        print(f"  chunk {i}: {count_tokens(c)} tokens")

    print("\n=== Table chunking: block index 9 (first table) ===")
    header_line = "Apple Inc. | Item 7 Table | Columns: 2024, Change, 2023"
    table_chunks = chunk_table_block(blocks[9], header_line)
    print(f"Produced {len(table_chunks)} chunks")
    for i, c in enumerate(table_chunks):
        print(f"  chunk {i}: {count_tokens(c)} tokens")
        
    print("\n=== Boundary check: end of chunk 0 vs start of chunk 1 ===")
    print("END OF CHUNK 0:")
    print(repr(prose_chunks[0][-150:]))
    print("\nSTART OF CHUNK 1:")
    print(repr(prose_chunks[1][:150]))