import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "data" / "clubs.json"

API_URL = "https://terplink.umd.edu/api/discovery/search/organizations"

STUDENT_ORG_BRANCH_ID = 226014
PAGE_SIZE = 100


IGNORED_CATEGORY_NAMES = {
    "SGA Recognized",
    "KFS Accounts",
}


BLOCKED_CATEGORY_NAMES = {
    "Univ. Department",
}


def clean_html(html: str | None) -> str:
    if not html:
        return ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    return " ".join(
        soup.get_text(
            " ",
            strip=True
        ).split()
    )


def clean_text(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(
        text.split()
    )


def is_internal_category(
    category: str
) -> bool:

    cleaned = clean_text(
        category
    )

    upper_cleaned = cleaned.upper()

    return (
        upper_cleaned.startswith("ADM_")
        or upper_cleaned.startswith("FROZEN")
        or cleaned in IGNORED_CATEGORY_NAMES
    )


def contains_term(
    text: str,
    term: str
) -> bool:

    if not text or not term:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(term.lower())
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        text.lower(),
        flags=re.IGNORECASE
    ) is not None


def choose_category(
    category_names: list[str]
) -> str:

    for category in category_names:

        cleaned = clean_text(
            category
        )

        if not cleaned:
            continue

        if is_internal_category(
            cleaned
        ):
            continue

        if cleaned in BLOCKED_CATEGORY_NAMES:
            continue

        return cleaned

    return "General Interest"


def build_tags(
    name: str,
    description: str,
    summary: str,
    categories: list[str],
) -> list[str]:

    text = (
        f"{name} "
        f"{description} "
        f"{summary}"
    ).lower()

    possible_tags = [
        "programming",
        "software",
        "computer science",
        "engineering",
        "robotics",
        "electronics",
        "cybersecurity",
        "artificial intelligence",
        "machine learning",
        "data science",
        "open source",
        "research",

        "business",
        "finance",
        "investment",
        "investing",
        "marketing",
        "entrepreneurship",
        "consulting",
        "product management",

        "medicine",
        "healthcare",
        "health",
        "biology",
        "chemistry",
        "physics",
        "psychology",
        "neuroscience",

        "leadership",
        "professional development",
        "networking",
        "community service",
        "volunteering",

        "activism",
        "advocacy",

        "music",
        "dance",
        "art",
        "theatre",
        "theater",
        "film",
        "photography",

        "culture",
        "religion",

        "sports",
        "fitness",
        "recreation",
        "soccer",
        "basketball",
        "football",
        "tennis",
        "running",
        "skiing",
        "snowboarding",

        "gaming",
        "esports",
    ]

    tags = []

    for tag in possible_tags:

        if contains_term(
            text,
            tag
        ):
            tags.append(
                tag
            )

    for category in categories:

        cleaned_category = clean_text(
            category
        )

        if not cleaned_category:
            continue

        if is_internal_category(
            cleaned_category
        ):
            continue

        if (
            cleaned_category
            in BLOCKED_CATEGORY_NAMES
        ):
            continue

        already_exists = any(
            existing.lower()
            == cleaned_category.lower()
            for existing in tags
        )

        if not already_exists:
            tags.append(
                cleaned_category
            )

    return tags[:12]


def fetch_organizations() -> list[dict]:
    organizations = []
    skip = 0

    print(
        "Downloading all TerpLink organizations..."
    )

    while True:

        params = {
            "orderBy[0]": "UpperName asc",
            "top": PAGE_SIZE,
            "filter": "",
            "query": "",
            "skip": skip,
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30,
            headers={
                "User-Agent":
                    "UMD-Campus-Opportunity-Finder/1.0"
            },
        )

        response.raise_for_status()

        data = response.json()

        results = data.get(
            "value",
            []
        )

        if not results:
            break

        print(
            f"Fetched {len(results)} records "
            f"starting at {skip}"
        )

        organizations.extend(
            results
        )

        skip += PAGE_SIZE

        if len(results) < PAGE_SIZE:
            break

        time.sleep(
            0.2
        )

    print()

    print(
        f"Downloaded {len(organizations)} "
        f"total TerpLink records."
    )

    return organizations


def is_useful_student_org(
    organization: dict
) -> bool:

    if (
        organization.get("BranchId")
        != STUDENT_ORG_BRANCH_ID
    ):
        return False

    if (
        organization.get("Status")
        != "Active"
    ):
        return False

    if (
        organization.get("Visibility")
        != "Public"
    ):
        return False

    categories = (
        organization.get(
            "CategoryNames"
        )
        or []
    )

    cleaned_categories = {
        clean_text(category)
        for category in categories
    }

    if (
        cleaned_categories
        & BLOCKED_CATEGORY_NAMES
    ):
        return False

    name = clean_text(
        organization.get(
            "Name"
        )
    )

    website_key = clean_text(
        organization.get(
            "WebsiteKey"
        )
    )

    if not name:
        return False

    if not website_key:
        return False

    description = clean_html(
        organization.get(
            "Description"
        )
    )

    summary = clean_text(
        organization.get(
            "Summary"
        )
    )

    useful_text = (
        description
        + " "
        + summary
    ).strip()

    if len(useful_text) < 40:
        return False

    return True


def convert_to_club(
    organization: dict,
    club_id: int,
) -> dict:

    name = clean_text(
        organization.get(
            "Name"
        )
    )

    website_key = clean_text(
        organization.get(
            "WebsiteKey"
        )
    )

    description = clean_html(
        organization.get(
            "Description"
        )
    )

    summary = clean_text(
        organization.get(
            "Summary"
        )
    )

    if not description:
        description = summary

    category_names = (
        organization.get(
            "CategoryNames"
        )
        or []
    )

    category = choose_category(
        category_names
    )

    tags = build_tags(
        name,
        description,
        summary,
        category_names,
    )

    url = (
        "https://terplink.umd.edu/"
        f"organization/{website_key}"
    )

    return {
        "id": club_id,
        "name": name,
        "description": description,
        "category": category,
        "tags": tags,
        "url": url,
    }


def main():
    organizations = (
        fetch_organizations()
    )

    clubs = []
    skipped = 0

    for organization in organizations:

        if not is_useful_student_org(
            organization
        ):
            skipped += 1
            continue

        club = convert_to_club(
            organization,
            len(clubs) + 1,
        )

        clubs.append(
            club
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            clubs,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("DATASET COMPLETE")

    print(
        f"Useful student organizations: "
        f"{len(clubs)}"
    )

    print(
        f"Filtered records: {skipped}"
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print()
    print("Example organizations:")

    for club in clubs[:10]:

        print(
            f"- {club['name']} "
            f"({club['category']})"
        )


if __name__ == "__main__":
    main()