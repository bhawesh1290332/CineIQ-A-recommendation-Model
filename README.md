# CineIQ-A-recommendation-Model
We built a recommendation model based on hybrid scoring of SVD and TF-IDF cosine similarity followed by DistilBERT reranking and explanatory terms using LIME.

# CineIQ 
> Personalized movie recommendations powered by SVD + TF-IDF + DistilBERT + LIME

---

##  Local Setup (Windows)

### Step 1 — Create a virtual environment
```bash
python -m venv venv
```

### Step 2 — Activate
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

### Step 3 — install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — run App.py
```bash
streamlit run APP.py
```

> Browser will automatically open at - `http://localhost:8501` 

---

##  Pipeline

For the detailed explanation refer to CineIQ_Explanatory_Doc.pdf. Theoretical aspect
of the project has been covered in that pdf file.

```
User Input (User ID / Genre)
        │
        ▼
┌─────────────────────┐
│   SVD (Collaborative │  - trained from ratings.csv
│   Filtering)         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF (Content     │  - genres + keywords + companies
│  Based Filtering)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Ensemble (Top 50)   │  - SVD 80% + TF-IDF 20%
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  DistilBERT Rerank   │  - sentiment from TMDB reviews
│  (Top 10)            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LIME Explanation    │  ← Why this movie?
└────────┬────────────┘
         │
         ▼
    Final Top 10
  Recommendations
```

---

##  Required Files

these files should be present in project folder:

```
CineIQ/
├── APP.py
├── CineIQ1.ipynb
├── requirements_cineiq.txt
├── tmdb.env file which contains your TMDB token for fetching reviews (example file is given just paste your token there) 
├── ratings.csv
├── movies_metadata.csv
├── links.csv
├── svd_model.pkl
├── reviews_cache.db
├── IMDB Dataset.csv
└── distilbert-sentiment-final/


distilbert-sentiment-final/, svd_model.pkl, and reviews_cache.db files will be obtained on running the CineIQ1.ipynb,
for fetching reviews from TMBD site it will take somewhere around 2.5 hours since it is obtaining close to 10k reviews,
the SVD will take 4-5 minutes to train and if you use your GPU for training distilbert it can reduce the training time
from 40-45 minutes to somewhere around 10-15 minutes. My laptop has RTX 4060 so the time got reduced to 12 minutes.

The csv files links.csv and ratings.csv are in the movielens25M data zip file and the tmdb_movie_dataset has been
renamed to movies_metadata.csv, and IMDB 50K reviews dataset has been used to finetune distilBERT. The datasets
are the ones stated in the problem statement and are readily available online.
```

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| Collaborative Filtering | SVD (scikit-surprise) |
| Content Based | TF-IDF (scikit-learn) |
| Sentiment Reranking | DistilBERT (HuggingFace) |
| Explainability | LIME |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
