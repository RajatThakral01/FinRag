SHORT_TO_FULL = {
    "Apple": "Apple Inc.",
    "Microsoft": "Microsoft Corporation",
    "Amazon": "Amazon.com Inc.",
    "NVIDIA": "NVIDIA Corporation",
    "Tesla": "Tesla Inc.",
    "Meta": "Meta Platforms Inc.",
    "Alphabet": "Alphabet Inc.",
    "Netflix": "Netflix Inc.",
    "Adobe": "Adobe Inc.",
}


def get_all_full_names() -> list[str]:
    return list(SHORT_TO_FULL.values())