import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CLUBS_FILE = BASE_DIR / "data" / "clubs.json"
OUTPUT_FILE = BASE_DIR / "data" / "club_embeddings.npy"


def build_weighted_club_text(club: dict) -> str:
    tags = club.get("tags", []) or []

    # Repeat identifying fields to give them more emphasis than the description.
    return " ".join(
        [
            club.get("name", ""),
            club.get("name", ""),
            club.get("name", ""),
            club.get("category", ""),
            club.get("category", ""),
            *tags,
            *tags,
            club.get("description", ""),
        ]
    )


def main() -> None:
    with open(CLUBS_FILE, "r", encoding="utf-8") as file:
        clubs = json.load(file)

    club_texts = [
        build_weighted_club_text(club)
        for club in clubs
    ]

    print(f"Loaded {len(clubs)} clubs.")
    print("Loading all-MiniLM-L6-v2 locally...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Generating normalized club embeddings...")

    embeddings = model.encode(
        club_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(OUTPUT_FILE, embeddings)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print(f"Saved embeddings to: {OUTPUT_FILE}")
    print(f"Shape: {embeddings.shape}")
    print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
