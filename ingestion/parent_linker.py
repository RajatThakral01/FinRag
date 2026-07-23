from itertools import groupby


def link_parent_chunks(chunks: list[dict]) -> list[dict]:
    grouped_by_block = [
        list(group) for _, group in groupby(chunks, key=lambda c: c["block_idx"])
    ]

    for i in range(1, len(grouped_by_block)):
        prev_block_chunks = grouped_by_block[i - 1]
        curr_block_chunks = grouped_by_block[i]

        last_of_prev = prev_block_chunks[-1]
        first_of_curr = curr_block_chunks[0]

        same_section = (
            last_of_prev["section_name"] == first_of_curr["section_name"]
            and last_of_prev["item_number"] == first_of_curr["item_number"]
        )
        different_type = last_of_prev["chunk_type"] != first_of_curr["chunk_type"]

        if same_section and different_type:
            first_of_curr["parent_chunk_id"] = last_of_prev["chunk_id"]

    return chunks