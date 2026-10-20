# UMD Opportunity Finder

A UMD student organization finder that uses semantic embeddings and hybrid relevance scoring to recommend clubs based on a student's interests, skills, major, and class year.

## Live Demo

https://umd-opportunity-finder.onrender.com

## What It Does

UMD Opportunity Finder helps University of Maryland students discover student organizations that match their profile. Users enter their major, class year, interests, and skills, and the app ranks relevant organizations from a dataset of roughly 900 active public TerpLink organizations.

The recommendation system combines multiple signals:

- **55% Semantic similarity** using `all-MiniLM-L6-v2` embeddings
- **25% Interest matching**
- **10% Major relevance**
- **10% Skill matching**
- Additional boosts for organization-name matches and class-year eligibility

## Architecture

### Frontend

- React
- TypeScript
- Vite
- Transformers.js
- Browser-side semantic embedding generation

### Backend

- Python
- FastAPI
- NumPy
- Precomputed organization embeddings

### Deployment

- Frontend: Render Static Site
- Backend: Render Web Service

## How Semantic Matching Works

To keep the backend lightweight enough for a low-memory deployment, organization embeddings are generated ahead of time and stored in `backend/data/club_embeddings.npy`.

When a user submits a profile:

1. The browser builds a semantic profile string from the student's major, class year, interests, and skills.
2. Transformers.js generates a 384-dimensional embedding using `Xenova/all-MiniLM-L6-v2`.
3. The frontend sends the embedding and profile data to the FastAPI backend.
4. The backend compares the student's vector against the precomputed organization embeddings using cosine similarity.
5. Semantic similarity is combined with interest, major, skill, name-match, and year-eligibility signals.
6. The top 20 organizations are returned and displayed to the user.

## Dataset

The project uses publicly available TerpLink organization data.

The data pipeline:

- Retrieves about 1,043 organization records from TerpLink's public organization API
- Filters for active, public student organizations
- Removes internal or administrative categories
- Cleans HTML from descriptions
- Produces a final dataset of about 904 organizations

The processed data is stored in:

```text
backend/data/clubs.json
```

Precomputed semantic embeddings are stored in:

```text
backend/data/club_embeddings.npy
```

## Project Structure

```text
CampusOpportunityFinder/
├── backend/
│   ├── data/
│   │   ├── clubs.json
│   │   └── club_embeddings.npy
│   ├── generate_embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── scraper.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## Running Locally

### Backend

From the project root:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

### Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend normally runs at:

```text
http://localhost:5173
```

## Regenerating Organization Embeddings

If `clubs.json` changes, regenerate the embeddings locally before deploying:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python generate_embeddings.py
```

This recreates:

```text
backend/data/club_embeddings.npy
```

## API Endpoints

```text
GET  /
GET  /health
GET  /clubs
GET  /clubs/search/{search_term}
GET  /clubs/{club_id}
POST /recommendations
```

## Why the Embedding Model Runs in the Browser

The first version loaded the embedding model directly on the backend, but the model runtime exceeded the memory limit of the deployment environment.

The current architecture generates the student's embedding in the browser while the backend only loads precomputed vectors. This significantly reduces backend memory usage while preserving semantic matching.

## Tech Stack

**Frontend:** React, TypeScript, Vite, Transformers.js  
**Backend:** Python, FastAPI, NumPy  
**ML/NLP:** MiniLM sentence embeddings, cosine similarity  
**Data:** TerpLink public organization data  
**Deployment:** Render

## Future Improvements

- Improve recommendation explanations
- Add more refined academic and career-interest signals
- Add saved profiles and favorite organizations
- Improve search and filtering
- Add organization popularity or activity signals
- Add analytics for recommendation quality

## Disclaimer

This project is an independent student project and is not affiliated with or officially endorsed by the University of Maryland or TerpLink.
