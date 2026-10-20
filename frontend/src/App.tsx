import { useMemo, useState } from "react";
import { pipeline } from "@huggingface/transformers";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type Club = {
  id: number;
  name: string;
  description: string;
  category: string;
  tags: string[];
  url: string;
};

type Recommendation = {
  club: Club;
  score: number;
  semantic_score: number;
  interest_score: number;
  major_score: number;
  skill_score: number;
  matched_terms: string[];
  explanation: string | null;
};


let embeddingPipelinePromise: Promise<any> | null = null;

function getEmbeddingPipeline() {
  if (!embeddingPipelinePromise) {
    embeddingPipelinePromise = pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );
  }

  return embeddingPipelinePromise;
}

const UMD_MAJORS = [
  "Accounting",
  "Aerospace Engineering",
  "African American and Africana Studies",
  "American Studies",
  "Anthropology",
  "Architecture",
  "Art",
  "Art History",
  "Astronomy",
  "Biochemistry",
  "Biological Sciences",
  "Biomedical Engineering",
  "Business Analytics",
  "Chemical Engineering",
  "Chemistry",
  "Civil Engineering",
  "Communication",
  "Computer Engineering",
  "Computer Science",
  "Criminology and Criminal Justice",
  "Dance",
  "Economics",
  "Electrical Engineering",
  "English",
  "Environmental Science",
  "Finance",
  "Geographical Sciences",
  "Government and Politics",
  "History",
  "Information Science",
  "Journalism",
  "Kinesiology",
  "Management",
  "Marketing",
  "Materials Science and Engineering",
  "Mathematics",
  "Mechanical Engineering",
  "Music",
  "Neuroscience",
  "Nursing",
  "Philosophy",
  "Physics",
  "Psychology",
  "Public Health Science",
  "Public Policy",
  "Sociology",
  "Spanish",
  "Statistics",
  "Supply Chain Management",
  "Theatre",
];

function App() {
  const [major, setMajor] = useState("");
  const [year, setYear] = useState("");
  const [interests, setInterests] = useState("");
  const [skills, setSkills] = useState("");

  const [recommendations, setRecommendations] =
    useState<Recommendation[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [category, setCategory] = useState("All");
  const [minimumScore, setMinimumScore] = useState(0);
  const [resultLimit, setResultLimit] = useState(10);
  const [clubSearch, setClubSearch] = useState("");

  async function findClubs() {
    setLoading(true);
    setError("");
    setRecommendations([]);

    try {
      const interestList = interests
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);

      const skillList = skills
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);

      const semanticText = [
        major,
        year,
        ...interestList,
        ...interestList,
        ...interestList,
        ...skillList,
        ...skillList,
      ].join(" ");

      const extractor = await getEmbeddingPipeline();

      const embeddingOutput = await extractor(
        semanticText,
        {
          pooling: "mean",
          normalize: true,
        }
      );

      const embedding = Array.from(
        embeddingOutput.data as Float32Array
      );

      const response = await fetch(
        `${API_BASE_URL}/recommendations`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            major,
            year,
            interests: interestList,
            skills: skillList,
            embedding,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          "Could not load recommendations."
        );
      }

      const data: Recommendation[] =
        await response.json();

      setRecommendations(data);
    } catch (err) {
      console.error(err);

      setError(
        "Could not connect to the recommendation service. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setMajor("");
    setYear("");
    setInterests("");
    setSkills("");

    setRecommendations([]);

    setCategory("All");
    setMinimumScore(0);
    setResultLimit(10);
    setClubSearch("");
    setError("");
  }

  const categories = useMemo(() => {
    const uniqueCategories = Array.from(
      new Set(
        recommendations.map(
          (recommendation) =>
            recommendation.club.category
        )
      )
    );

    return [
      "All",
      ...uniqueCategories.sort(),
    ];
  }, [recommendations]);

  const visibleRecommendations =
    useMemo(() => {
      return recommendations
        .filter((recommendation) => {
          const categoryMatches =
            category === "All" ||
            recommendation.club.category ===
              category;

          const scoreMatches =
            recommendation.score >=
            minimumScore;

          const searchMatches =
            recommendation.club.name
              .toLowerCase()
              .includes(
                clubSearch.toLowerCase()
              );

          return (
            categoryMatches &&
            scoreMatches &&
            searchMatches
          );
        })
        .slice(0, resultLimit);
    }, [
      recommendations,
      category,
      minimumScore,
      resultLimit,
      clubSearch,
    ]);

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-content">
          <div className="badge">
            University of Maryland
          </div>

          <h1>
            Find your place at{" "}
            <span>UMD.</span>
          </h1>

          <p>
            Discover student organizations
            that match your interests,
            skills, academic background, and
            goals using semantic embeddings
            and hybrid relevance scoring.
          </p>

          <div className="hero-stats">
            <div className="hero-stat">
              <strong>900+</strong>
              <span>Organizations</span>
            </div>

            <div className="hero-stat">
              <strong>4</strong>
              <span>Ranking Signals</span>
            </div>

            <div className="hero-stat">
              <strong>AI</strong>
              <span>Semantic Matching</span>
            </div>
          </div>
        </div>
      </header>

      <main className="container">
        <section className="search-panel">
          <div className="section-heading">
            <p className="eyebrow">
              YOUR PROFILE
            </p>

            <h2>
              What are you interested in?
            </h2>

            <p>
              Tell us about yourself and
              we'll rank UMD organizations
              based on your interests,
              skills, and academic
              background.
            </p>
          </div>

          <div className="form-grid">
            <div className="field">
              <label htmlFor="major">
                Major
              </label>

              <input
                id="major"
                list="major-options"
                value={major}
                onChange={(event) =>
                  setMajor(
                    event.target.value
                  )
                }
                placeholder="Search or select your major"
                autoComplete="off"
              />

              <datalist id="major-options">
                {UMD_MAJORS.map(
                  (majorName) => (
                    <option
                      key={majorName}
                      value={majorName}
                    />
                  )
                )}
              </datalist>

              <small>
                Start typing to search or
                choose from the suggested
                majors.
              </small>
            </div>

            <div className="field">
              <label htmlFor="year">
                Class Year
              </label>

              <select
                id="year"
                value={year}
                onChange={(event) =>
                  setYear(
                    event.target.value
                  )
                }
              >
                <option
                  value=""
                  disabled
                >
                  Select
                </option>

                <option>Freshman</option>
                <option>Sophomore</option>
                <option>Junior</option>
                <option>Senior</option>
                <option>Graduate</option>
              </select>
            </div>

            <div className="field full-width">
              <label htmlFor="interests">
                Interests
              </label>

              <input
                id="interests"
                value={interests}
                onChange={(event) =>
                  setInterests(
                    event.target.value
                  )
                }
                placeholder="music, finance, healthcare, sports, art, robotics, volunteering"
              />

              <small>
                Add anything you enjoy or
                want to explore. Separate
                interests with commas.
              </small>
            </div>

            <div className="field full-width">
              <label htmlFor="skills">
                Skills
              </label>

              <input
                id="skills"
                value={skills}
                onChange={(event) =>
                  setSkills(
                    event.target.value
                  )
                }
                placeholder="writing, public speaking, Excel, research, design, Python, leadership"
              />

              <small>
                Include technical, creative,
                academic, professional, or
                interpersonal skills.
                Separate them with commas.
              </small>
            </div>
          </div>

          <div className="action-row">
            <button
              className="find-button"
              onClick={findClubs}
              disabled={
                loading ||
                !major.trim() ||
                !year
              }
            >
              {loading
                ? "Finding Matches..."
                : "Find My Clubs"}
            </button>

            <button
              className="reset-button"
              onClick={resetForm}
              type="button"
            >
              Reset
            </button>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </section>

        <section className="how-it-works">
          <div className="section-heading">
            <p className="eyebrow">
              HOW MATCHING WORKS
            </p>

            <h2>
              More than just keyword search.
            </h2>

            <p>
              Each organization is compared
              against your profile using
              semantic AI matching plus
              structured relevance signals.
            </p>
          </div>

          <div className="weight-grid">
            <WeightCard
              number="55%"
              title="Semantic"
              text="AI embeddings measure overall similarity between your profile and each organization."
            />

            <WeightCard
              number="25%"
              title="Interests"
              text="Your interests are matched against related concepts, activities, and organization topics."
            />

            <WeightCard
              number="10%"
              title="Major"
              text="Your field of study provides another signal when an organization is academically relevant."
            />

            <WeightCard
              number="10%"
              title="Skills"
              text="Technical, creative, professional, academic, and interpersonal skills can strengthen a match."
            />
          </div>
        </section>

        {recommendations.length > 0 && (
          <section className="results">
            <div className="results-title-row">
              <div className="section-heading">
                <p className="eyebrow">
                  RECOMMENDATIONS
                </p>

                <h2>
                  Your best matches
                </h2>

                <p>
                  Ranked from more than 900
                  active UMD student
                  organizations.
                </p>
              </div>

              <div className="result-count">
                {
                  visibleRecommendations.length
                }{" "}
                shown
              </div>
            </div>

            <div className="filters">
              <div className="filter-field search-filter">
                <label>
                  Search Results
                </label>

                <input
                  value={clubSearch}
                  onChange={(event) =>
                    setClubSearch(
                      event.target.value
                    )
                  }
                  placeholder="Search organization name..."
                />
              </div>

              <div className="filter-field">
                <label>Category</label>

                <select
                  value={category}
                  onChange={(event) =>
                    setCategory(
                      event.target.value
                    )
                  }
                >
                  {categories.map(
                    (categoryName) => (
                      <option
                        key={
                          categoryName
                        }
                      >
                        {categoryName}
                      </option>
                    )
                  )}
                </select>
              </div>

              <div className="filter-field">
                <label>
                  Minimum Match
                </label>

                <select
                  value={minimumScore}
                  onChange={(event) =>
                    setMinimumScore(
                      Number(
                        event.target
                          .value
                      )
                    )
                  }
                >
                  <option value={0}>
                    Any Score
                  </option>
                  <option value={30}>
                    30%+
                  </option>
                  <option value={40}>
                    40%+
                  </option>
                  <option value={50}>
                    50%+
                  </option>
                  <option value={60}>
                    60%+
                  </option>
                </select>
              </div>

              <div className="filter-field">
                <label>Results</label>

                <select
                  value={resultLimit}
                  onChange={(event) =>
                    setResultLimit(
                      Number(
                        event.target
                          .value
                      )
                    )
                  }
                >
                  <option value={5}>
                    Top 5
                  </option>
                  <option value={10}>
                    Top 10
                  </option>
                  <option value={20}>
                    Top 20
                  </option>
                </select>
              </div>
            </div>

            <div className="results-grid">
              {visibleRecommendations.map(
                (
                  recommendation,
                  index
                ) => (
                  <article
                    className="club-card"
                    key={
                      recommendation
                        .club.id
                    }
                  >
                    <div className="club-top">
                      <div>
                        <span className="rank">
                          #{index + 1}
                        </span>

                        <span className="category">
                          {
                            recommendation
                              .club
                              .category
                          }
                        </span>
                      </div>

                      <div className="score">
                        {recommendation.score.toFixed(
                          1
                        )}
                        %
                        <span>
                          match
                        </span>
                      </div>
                    </div>

                    <h3>
                      {
                        recommendation
                          .club.name
                      }
                    </h3>

                    <p className="description">
                      {
                        recommendation
                          .club
                          .description
                      }
                    </p>

                    <div className="score-breakdown">
                      <ScoreItem
                        label="Semantic"
                        score={
                          recommendation.semantic_score
                        }
                      />

                      <ScoreItem
                        label="Interests"
                        score={
                          recommendation.interest_score
                        }
                      />

                      <ScoreItem
                        label="Major"
                        score={
                          recommendation.major_score
                        }
                      />

                      <ScoreItem
                        label="Skills"
                        score={
                          recommendation.skill_score
                        }
                      />
                    </div>

                    {recommendation.explanation && (
                      <div className="explanation">
                        <strong>
                          Why it matches
                        </strong>

                        <p>
                          {
                            recommendation.explanation
                          }
                        </p>
                      </div>
                    )}

                    {recommendation
                      .matched_terms
                      .length > 0 && (
                      <div className="matched-section">
                        <strong>
                          Matched profile
                          terms
                        </strong>

                        <div className="matched-tags">
                          {recommendation.matched_terms.map(
                            (term) => (
                              <span
                                key={
                                  term
                                }
                              >
                                {
                                  term
                                }
                              </span>
                            )
                          )}
                        </div>
                      </div>
                    )}

                    <div className="tags">
                      {recommendation.club.tags.map(
                        (tag) => (
                          <span
                            key={tag}
                          >
                            {tag}
                          </span>
                        )
                      )}
                    </div>

                    <a
                      className="club-link"
                      href={
                        recommendation
                          .club.url
                      }
                      target="_blank"
                      rel="noreferrer"
                    >
                      View on TerpLink →
                    </a>
                  </article>
                )
              )}
            </div>

            {visibleRecommendations.length ===
              0 && (
              <div className="no-results">
                No organizations match
                your current filters.
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

function ScoreItem({
  label,
  score,
}: {
  label: string;
  score: number;
}) {
  return (
    <div className="score-item">
      <div className="score-item-top">
        <span>{label}</span>

        <strong>
          {score.toFixed(0)}%
        </strong>
      </div>

      <div className="score-bar">
        <div
          className="score-bar-fill"
          style={{
            width: `${Math.min(
              score,
              100
            )}%`,
          }}
        />
      </div>
    </div>
  );
}

function WeightCard({
  number,
  title,
  text,
}: {
  number: string;
  title: string;
  text: string;
}) {
  return (
    <div className="weight-card">
      <div className="weight-number">
        {number}
      </div>

      <h3>{title}</h3>

      <p>{text}</p>
    </div>
  );
}

export default App;