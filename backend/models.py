from pydantic import BaseModel


class Club(BaseModel):
    id: int
    name: str
    description: str
    category: str
    tags: list[str]
    url: str


class StudentProfile(BaseModel):
    major: str
    year: str
    interests: list[str]
    skills: list[str]


class ClubRecommendation(BaseModel):
    club: Club
    score: float
    semantic_score: float
    interest_score: float
    major_score: float
    skill_score: float
    matched_terms: list[str]
    explanation: str | None = None