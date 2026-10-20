import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from models import Club, StudentProfile, ClubRecommendation


app = FastAPI(
    title="UMD Campus Opportunity Finder API",
    description="AI-powered backend for discovering UMD student organizations.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://umd-opportunity-finder.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
CLUBS_FILE = BASE_DIR / "data" / "clubs.json"
EMBEDDINGS_FILE = BASE_DIR / "data" / "club_embeddings.npy"


class RecommendationRequest(StudentProfile):
    embedding: list[float]


SYNONYM_GROUPS = {
    "cars": [
        "car",
        "cars",
        "automotive",
        "vehicle",
        "vehicles",
        "racing",
        "race car",
        "formula sae",
        "baja sae",
        "motorsport",
        "electric vehicle",
    ],

    "rockets": [
        "rocket",
        "rockets",
        "rocketry",
        "rocket engine",
        "propulsion",
        "launch vehicle",
        "aerospace",
    ],

    "coding": [
        "coding",
        "programming",
        "software",
        "software development",
        "computer science",
        "developer",
    ],

    "programming": [
        "coding",
        "programming",
        "software",
        "software development",
        "computer science",
    ],

    "ai": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural network",
    ],

    "cybersecurity": [
        "cybersecurity",
        "cyber security",
        "information security",
        "network security",
        "ethical hacking",
    ],

    "robotics": [
        "robotics",
        "robot",
        "robots",
        "automation",
        "autonomous",
        "mechatronics",
    ],

    "electronics": [
        "electronics",
        "electrical",
        "circuits",
        "embedded",
        "embedded systems",
        "microcontroller",
        "arduino",
    ],

    "investing": [
        "investing",
        "investment",
        "investments",
        "stocks",
        "equities",
        "portfolio",
        "asset management",
        "capital markets",
        "financial markets",
    ],

    "finance": [
        "finance",
        "financial",
        "investment",
        "investing",
        "banking",
        "capital markets",
        "financial markets",
    ],

    "consulting": [
        "consulting",
        "consultant",
        "consultants",
        "advisory",
        "strategy consulting",
        "management consulting",
        "pro bono consulting",
    ],

    "business": [
        "business",
        "entrepreneurship",
        "startup",
        "management",
        "commerce",
    ],

    "medicine": [
        "medicine",
        "medical",
        "healthcare",
        "pre-med",
        "health",
    ],

    "music": [
        "music",
        "musician",
        "singing",
        "choir",
        "band",
        "instrument",
    ],

    "sports": [
        "sports",
        "athletics",
        "fitness",
        "recreation",
    ],

    "writing": [
        "writing",
        "journalism",
        "reporting",
        "publication",
        "media",
        "editorial",
    ],

    "volunteering": [
        "volunteer",
        "volunteering",
        "service",
        "community service",
        "outreach",
        "nonprofit",
    ],

    "design": [
        "design",
        "creative",
        "visual",
        "graphic design",
        "architecture",
        "art",
    ],
}


MAJOR_KEYWORDS = {
    "accounting": [
        "accounting",
        "audit",
        "tax",
        "financial",
        "finance",
        "business",
    ],

    "aerospace engineering": [
        "aerospace",
        "rocket",
        "rocketry",
        "aviation",
        "aircraft",
        "propulsion",
        "engineering",
        "space",
    ],

    "african american and africana studies": [
        "african american",
        "africana",
        "culture",
        "history",
        "social justice",
        "community",
        "advocacy",
    ],

    "american studies": [
        "american studies",
        "culture",
        "history",
        "society",
        "politics",
        "media",
    ],

    "anthropology": [
        "anthropology",
        "culture",
        "society",
        "human behavior",
        "archaeology",
        "research",
    ],

    "architecture": [
        "architecture",
        "design",
        "built environment",
        "urban",
        "planning",
        "construction",
    ],

    "art": [
        "art",
        "creative",
        "visual arts",
        "design",
        "painting",
        "drawing",
        "gallery",
    ],

    "art history": [
        "art",
        "history",
        "museum",
        "culture",
        "visual arts",
        "research",
    ],

    "astronomy": [
        "astronomy",
        "space",
        "astrophysics",
        "physics",
        "telescope",
        "research",
    ],

    "biochemistry": [
        "biochemistry",
        "biology",
        "chemistry",
        "molecular",
        "research",
        "health",
    ],

    "biological sciences": [
        "biology",
        "biological",
        "life science",
        "ecology",
        "medicine",
        "health",
        "research",
    ],

    "biomedical engineering": [
        "biomedical",
        "engineering",
        "medical",
        "healthcare",
        "prosthetics",
        "biology",
        "technology",
    ],

    "business analytics": [
        "business analytics",
        "analytics",
        "data",
        "business",
        "statistics",
        "decision making",
    ],

    "chemical engineering": [
        "chemical engineering",
        "chemistry",
        "engineering",
        "process",
        "materials",
        "energy",
    ],

    "chemistry": [
        "chemistry",
        "chemical",
        "laboratory",
        "materials",
        "research",
        "science",
    ],

    "civil engineering": [
        "civil engineering",
        "engineering",
        "infrastructure",
        "construction",
        "transportation",
        "environmental",
    ],

    "communication": [
        "communication",
        "media",
        "public relations",
        "public speaking",
        "marketing",
        "journalism",
    ],

    "computer engineering": [
        "computer engineering",
        "engineering",
        "electronics",
        "electrical",
        "embedded",
        "hardware",
        "software",
        "programming",
        "robotics",
        "computer science",
        "technology",
    ],

    "computer science": [
        "computer science",
        "software",
        "programming",
        "coding",
        "technology",
        "data",
        "artificial intelligence",
        "cybersecurity",
    ],

    "criminology and criminal justice": [
        "criminology",
        "criminal justice",
        "law",
        "justice",
        "security",
        "policy",
        "advocacy",
    ],

    "dance": [
        "dance",
        "performance",
        "performing arts",
        "movement",
        "choreography",
        "arts",
    ],

    "economics": [
        "economics",
        "economy",
        "markets",
        "finance",
        "policy",
        "data",
        "research",
    ],

    "electrical engineering": [
        "electrical",
        "electronics",
        "circuits",
        "engineering",
        "hardware",
        "embedded",
        "robotics",
    ],

    "english": [
        "english",
        "writing",
        "literature",
        "publishing",
        "creative writing",
        "journalism",
    ],

    "environmental science": [
        "environmental",
        "sustainability",
        "climate",
        "ecology",
        "conservation",
        "science",
    ],

    "finance": [
        "finance",
        "financial",
        "investment",
        "investing",
        "portfolio",
        "banking",
        "capital markets",
        "financial markets",
        "stocks",
        "equities",
        "business",
    ],

    "geographical sciences": [
        "geography",
        "geospatial",
        "gis",
        "mapping",
        "environment",
        "urban",
    ],

    "government and politics": [
        "government",
        "politics",
        "policy",
        "public policy",
        "advocacy",
        "debate",
        "international relations",
    ],

    "history": [
        "history",
        "historical",
        "museum",
        "culture",
        "archives",
        "research",
    ],

    "information science": [
        "information science",
        "data",
        "technology",
        "information",
        "analytics",
        "user experience",
        "digital",
    ],

    "journalism": [
        "journalism",
        "media",
        "writing",
        "reporting",
        "news",
        "communications",
        "publications",
    ],

    "kinesiology": [
        "kinesiology",
        "fitness",
        "sports",
        "health",
        "exercise",
        "wellness",
    ],

    "management": [
        "management",
        "business",
        "leadership",
        "operations",
        "entrepreneurship",
        "consulting",
    ],

    "marketing": [
        "marketing",
        "advertising",
        "branding",
        "social media",
        "business",
        "public relations",
    ],

    "materials science and engineering": [
        "materials",
        "engineering",
        "manufacturing",
        "nanotechnology",
        "chemistry",
        "research",
    ],

    "mathematics": [
        "mathematics",
        "math",
        "statistics",
        "quantitative",
        "analytics",
        "research",
    ],

    "mechanical engineering": [
        "mechanical",
        "engineering",
        "robotics",
        "automotive",
        "vehicle",
        "manufacturing",
        "aerospace",
    ],

    "music": [
        "music",
        "musician",
        "performance",
        "band",
        "choir",
        "instrument",
        "audio",
    ],

    "neuroscience": [
        "neuroscience",
        "brain",
        "cognition",
        "biology",
        "psychology",
        "research",
        "health",
    ],

    "nursing": [
        "nursing",
        "healthcare",
        "health",
        "medical",
        "patient care",
        "community health",
    ],

    "philosophy": [
        "philosophy",
        "ethics",
        "logic",
        "debate",
        "critical thinking",
        "humanities",
    ],

    "physics": [
        "physics",
        "science",
        "astronomy",
        "engineering",
        "research",
        "quantum",
    ],

    "psychology": [
        "psychology",
        "mental health",
        "behavior",
        "cognition",
        "neuroscience",
        "research",
        "wellness",
    ],

    "public health science": [
        "public health",
        "health",
        "healthcare",
        "community health",
        "medicine",
        "wellness",
        "advocacy",
    ],

    "public policy": [
        "public policy",
        "policy",
        "government",
        "advocacy",
        "politics",
        "community",
    ],

    "sociology": [
        "sociology",
        "society",
        "social issues",
        "community",
        "culture",
        "research",
        "advocacy",
    ],

    "spanish": [
        "spanish",
        "language",
        "culture",
        "international",
        "latin america",
        "communication",
    ],

    "statistics": [
        "statistics",
        "data",
        "analytics",
        "quantitative",
        "research",
        "mathematics",
    ],

    "supply chain management": [
        "supply chain",
        "logistics",
        "operations",
        "business",
        "transportation",
        "management",
    ],

    "theatre": [
        "theatre",
        "theater",
        "acting",
        "performance",
        "performing arts",
        "production",
    ],
}


def load_clubs() -> list[Club]:

    with open(
        CLUBS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    return [
        Club(**club)
        for club in data
    ]


def normalize_text(
    text: str
) -> str:

    return text.strip().lower()


def contains_term(
    text: str,
    term: str
) -> bool:

    text = text.lower()
    term = term.lower().strip()

    if not term:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(term)
        + r"(?![a-z0-9])"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE
    ) is not None


def build_club_text(
    club: Club
) -> str:

    return " ".join(
        [
            club.name,
            club.description,
            club.category,
            *club.tags
        ]
    ).lower()


def build_weighted_club_text(
    club: Club
) -> str:

    return " ".join(
        [
            club.name,
            club.name,
            club.name,
            club.category,
            club.category,
            *club.tags,
            *club.tags,
            club.description,
        ]
    )


def build_student_semantic_text(
    profile: StudentProfile
) -> str:

    return " ".join(
        [
            profile.major,
            profile.year,
            *profile.interests,
            *profile.interests,
            *profile.interests,
            *profile.skills,
            *profile.skills,
        ]
    )


def expand_term(
    term: str
) -> list[str]:

    normalized = normalize_text(
        term
    )

    expanded = {
        normalized
    }

    for group_name, related_terms in (
        SYNONYM_GROUPS.items()
    ):

        group = [
            normalize_text(
                group_name
            ),
            *[
                normalize_text(item)
                for item in related_terms
            ]
        ]

        if normalized in group:

            expanded.update(
                group
            )

    return list(
        expanded
    )


def calculate_interest_score(
    profile: StudentProfile,
    club: Club
) -> tuple[float, list[str]]:

    if not profile.interests:
        return 0.0, []

    club_text = build_club_text(
        club
    )

    matched_interests = []

    for interest in profile.interests:

        candidates = expand_term(
            interest
        )

        matched = any(
            contains_term(
                club_text,
                candidate
            )
            for candidate in candidates
        )

        if matched:

            matched_interests.append(
                normalize_text(
                    interest
                )
            )

    score = (
        len(
            set(
                matched_interests
            )
        )
        / len(
            profile.interests
        )
    ) * 100

    return (
        round(
            score,
            1
        ),
        sorted(
            set(
                matched_interests
            )
        )
    )


def calculate_skill_score(
    profile: StudentProfile,
    club: Club
) -> tuple[float, list[str]]:

    if not profile.skills:
        return 0.0, []

    club_text = build_club_text(
        club
    )

    matched_skills = []

    for skill in profile.skills:

        candidates = expand_term(
            skill
        )

        matched = any(
            contains_term(
                club_text,
                candidate
            )
            for candidate in candidates
        )

        if matched:

            matched_skills.append(
                normalize_text(
                    skill
                )
            )

    score = (
        len(
            set(
                matched_skills
            )
        )
        / len(
            profile.skills
        )
    ) * 100

    return (
        round(
            score,
            1
        ),
        sorted(
            set(
                matched_skills
            )
        )
    )


def calculate_major_score(
    profile: StudentProfile,
    club: Club
) -> float:

    major = normalize_text(
        profile.major
    )

    club_text = build_club_text(
        club
    )

    if (
        major
        and contains_term(
            club_text,
            major
        )
    ):
        return 100.0

    related_terms = (
        MAJOR_KEYWORDS.get(
            major,
            []
        )
    )

    if not related_terms:
        return 0.0

    matched_terms = [
        term
        for term in related_terms
        if contains_term(
            club_text,
            term
        )
    ]

    match_count = len(
        matched_terms
    )

    if match_count == 0:
        return 0.0

    if match_count == 1:
        return 35.0

    if match_count == 2:
        return 60.0

    if match_count == 3:
        return 80.0

    return 100.0


def is_graduate_organization(
    club: Club
) -> bool:

    graduate_label = (
        "graduate student organizations"
    )

    if (
        normalize_text(
            club.category
        )
        == graduate_label
    ):
        return True

    return any(
        normalize_text(tag)
        == graduate_label
        for tag in club.tags
    )


def is_eligible_for_year(
    profile: StudentProfile,
    club: Club
) -> bool:

    year = normalize_text(
        profile.year
    )

    if (
        year != "graduate"
        and is_graduate_organization(
            club
        )
    ):
        return False

    return True


def calculate_year_boost(
    profile: StudentProfile,
    club: Club
) -> float:

    year = normalize_text(
        profile.year
    )

    if (
        year == "graduate"
        and is_graduate_organization(
            club
        )
    ):
        return 5.0

    return 0.0


def calculate_name_boost(
    profile: StudentProfile,
    club: Club
) -> float:

    boost = 0.0

    club_name = (
        club.name.lower()
    )

    for interest in (
        profile.interests
    ):

        candidates = expand_term(
            interest
        )

        if any(
            contains_term(
                club_name,
                candidate
            )
            for candidate in candidates
        ):
            boost += 8.0

    return min(
        boost,
        16.0
    )


def generate_explanation(
    interest_matches: list[str],
    skill_matches: list[str],
    semantic_score: float,
) -> str:

    all_matches = list(
        dict.fromkeys(
            interest_matches
            + skill_matches
        )
    )

    if all_matches:

        terms = ", ".join(
            all_matches[:4]
        )

        return (
            f"This organization aligns with your "
            f"profile through {terms}, along with "
            f"semantic similarity to your broader interests."
        )

    if semantic_score >= 60:

        return (
            "This organization is strongly related "
            "to your overall interests and background "
            "based on semantic similarity."
        )

    if semantic_score >= 45:

        return (
            "This organization has moderate similarity "
            "to your interests and may be worth exploring."
        )

    return (
        "This organization has some similarity to "
        "your profile, but it may be a weaker match."
    )


CLUBS = load_clubs()


CLUB_TEXTS = [
    build_weighted_club_text(
        club
    )
    for club in CLUBS
]


print(
    f"Loading precomputed embeddings for "
    f"{len(CLUBS)} UMD organizations..."
)


if not EMBEDDINGS_FILE.exists():
    raise RuntimeError(
        "Missing data/club_embeddings.npy. "
        "Run generate_embeddings.py locally before deployment."
    )


CLUB_EMBEDDINGS = np.load(
    EMBEDDINGS_FILE,
    mmap_mode="r"
)


if CLUB_EMBEDDINGS.shape[0] != len(CLUBS):
    raise RuntimeError(
        "club_embeddings.npy does not match clubs.json. "
        "Regenerate embeddings locally."
    )


CLUB_EMBEDDING_NORMS = np.linalg.norm(
    CLUB_EMBEDDINGS,
    axis=1
)


print(
    "Precomputed club embeddings ready."
)


@app.get("/")
def root():

    return {
        "message":
            "UMD Campus Opportunity Finder backend is running",
        "organizations":
            len(CLUBS)
    }


@app.get("/health")
def health():

    return {
        "status":
            "ok",
        "organizations":
            len(CLUBS)
    }


@app.get(
    "/clubs",
    response_model=list[Club]
)
def get_clubs():

    return CLUBS


@app.get(
    "/clubs/search/{search_term}",
    response_model=list[Club]
)
def search_clubs(
    search_term: str
):

    term = normalize_text(
        search_term
    )

    return [
        club
        for club in CLUBS
        if (
            term
            in club.name.lower()
            or term
            in club.description.lower()
        )
    ][:20]


@app.get(
    "/clubs/{club_id}",
    response_model=Club
)
def get_club(
    club_id: int
):

    for club in CLUBS:

        if club.id == club_id:
            return club

    raise HTTPException(
        status_code=404,
        detail="Club not found"
    )


@app.post(
    "/recommendations",
    response_model=list[ClubRecommendation]
)
def get_recommendations(
    profile: RecommendationRequest
):

    student_embedding = np.asarray(
        profile.embedding,
        dtype=np.float32
    )

    if student_embedding.ndim != 1:
        raise HTTPException(
            status_code=400,
            detail="Embedding must be one-dimensional."
        )

    if student_embedding.shape[0] != CLUB_EMBEDDINGS.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected embedding dimension {CLUB_EMBEDDINGS.shape[1]}, "
                f"received {student_embedding.shape[0]}."
            )
        )

    student_norm = np.linalg.norm(
        student_embedding
    )

    if student_norm == 0:
        raise HTTPException(
            status_code=400,
            detail="Embedding cannot be the zero vector."
        )

    similarities = (
        CLUB_EMBEDDINGS
        @ student_embedding
    ) / (
        CLUB_EMBEDDING_NORMS
        * student_norm
        + 1e-8
    )

    recommendations = []

    for index, club in enumerate(
        CLUBS
    ):

        if not is_eligible_for_year(
            profile,
            club
        ):
            continue

        raw_similarity = float(
            similarities[index]
        )

        semantic_score = max(
            0.0,
            min(
                100.0,
                raw_similarity * 100
            )
        )

        (
            interest_score,
            interest_matches
        ) = calculate_interest_score(
            profile,
            club
        )

        (
            skill_score,
            skill_matches
        ) = calculate_skill_score(
            profile,
            club
        )

        major_score = (
            calculate_major_score(
                profile,
                club
            )
        )

        final_score = (
            semantic_score * 0.55
            + interest_score * 0.25
            + major_score * 0.10
            + skill_score * 0.10
        )

        final_score += (
            calculate_name_boost(
                profile,
                club
            )
        )

        final_score += (
            calculate_year_boost(
                profile,
                club
            )
        )

        final_score = round(
            min(
                final_score,
                100
            ),
            1
        )

        matched_terms = sorted(
            set(
                interest_matches
                + skill_matches
            )
        )

        explanation = (
            generate_explanation(
                interest_matches,
                skill_matches,
                semantic_score
            )
        )

        recommendations.append(
            ClubRecommendation(
                club=club,
                score=final_score,
                semantic_score=round(
                    semantic_score,
                    1
                ),
                interest_score=interest_score,
                major_score=major_score,
                skill_score=skill_score,
                matched_terms=matched_terms,
                explanation=explanation
            )
        )

    recommendations.sort(
        key=lambda recommendation:
            recommendation.score,
        reverse=True
    )

    return recommendations[:20]