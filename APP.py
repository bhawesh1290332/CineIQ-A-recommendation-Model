# %%
# app.py  — save this in d:\Project&Learnings\CineIQ\

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import json
import re
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from surprise import SVD, Dataset, Reader

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineIQ", page_icon="🎬", layout="wide")

# ── Load all assets (cached so they only load once) ──────────────────────────
@st.cache_resource
def load_assets():
    # DataFrames
    ratings = pd.read_csv(os.path.join(BASE, 'ratings.csv'))
    links   = pd.read_csv(os.path.join(BASE, 'links.csv'))
    movies  = pd.read_csv(os.path.join(BASE, 'movies_metadata.csv'))

    # Clean and merge (same as notebook)
    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)
    merged = links.merge(movies, left_on='tmdbId', right_on='id', how='inner')
    merged['soup'] = (merged['genres'].fillna('') + ' ' +
                      merged['keywords'].fillna('') + ' ' +
                      merged['production_companies'].fillna(''))

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(merged['soup'])

    # SVD
    with open(os.path.join(BASE, 'svd_model.pkl'), 'rb') as f:
        svd = pickle.load(f)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()

    # DistilBERT
    model     = DistilBertForSequenceClassification.from_pretrained(
                    os.path.join(BASE, 'distilbert-sentiment-final'))
    tokenizer = DistilBertTokenizer.from_pretrained(
                    os.path.join(BASE, 'distilbert-sentiment-final'))
    model.eval()

    # Reviews cache
    conn = sqlite3.connect(os.path.join(BASE, 'reviews_cache.db'))
    rows = conn.execute('SELECT tmdb_id, reviews_json FROM reviews').fetchall()
    records = []
    for tmdb_id, reviews_json in rows:
        for review in json.loads(reviews_json):
            records.append({'tmdbId': tmdb_id, 'review': review})
    reviews_df = pd.DataFrame(records)

    return ratings, merged, tfidf, tfidf_matrix, svd, trainset, model, tokenizer, reviews_df

ratings, merged, tfidf, tfidf_matrix, svd, trainset, model, tokenizer, reviews_df = load_assets()

# ── Pipeline functions (same logic as notebook) ───────────────────────────────
CUSTOM_STOPWORDS = {
    'to','the','a','an','it','its','is','was','are','were','be','been',
    'being','have','has','had','do','does','did','will','would','shall',
    'should','may','might','must','can','could','i','me','my','we','our',
    'you','your','he','his','him','she','her','hers','they','them','their',
    'this','that','these','those','which','who','what','when','where','why',
    'how','all','each','every','both','few','more','most','other','some',
    'such','no','not','only','same','so','than','too','very','just','but',
    'and','or','if','of','at','by','for','with','about','into','through',
    'during','also','br','part','last','inter','because','even','never',
    'still','much','well','there','here','then','now','after','before'
}

def clean_review(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def svd_scores(user_id, movie_ids):
    return {mid: svd.predict(user_id, mid).est for mid in movie_ids}

def tfidf_scores(user_id, movie_ids, top_rated_movies):
    user_indices = [merged.index[merged['movieId'] == mid][0]
                    for mid in top_rated_movies
                    if len(merged.index[merged['movieId'] == mid]) > 0]
    if not user_indices:
        return {mid: 0.0 for mid in movie_ids}
    user_profile = np.asarray(tfidf_matrix[user_indices].mean(axis=0))
    candidate_indices, valid_ids = [], []
    for mid in movie_ids:
        match = merged.index[merged['movieId'] == mid]
        if len(match) > 0:
            candidate_indices.append(match[0])
            valid_ids.append(mid)
    if not candidate_indices:
        return {mid: 0.0 for mid in movie_ids}
    sims = cosine_similarity(user_profile, tfidf_matrix[candidate_indices])
    return {mid: float(sims[0][i]) for i, mid in enumerate(valid_ids)}

def ensemble_scores(svd_dict, tfidf_dict, w=0.65):
    all_ids   = set(svd_dict) & set(tfidf_dict)
    svd_vals  = np.array([svd_dict[m] for m in all_ids])
    tf_vals   = np.array([tfidf_dict[m] for m in all_ids])
    svd_norm  = (svd_vals - svd_vals.min()) / ((svd_vals.max() - svd_vals.min()) + 1e-9)
    tf_norm   = (tf_vals  - tf_vals.min())  / ((tf_vals.max()  - tf_vals.min())  + 1e-9)
    blended   = w * svd_norm + (1-w) * tf_norm
    scored    = dict(zip(all_ids, blended))
    return sorted(scored, key=scored.get, reverse=True)[:50]

def rerank_with_distilbert(top50_ids):
    scores = {}
    for mid in top50_ids:
        movie_reviews = reviews_df[reviews_df.tmdbId == mid]['review'].tolist()[:20]
        if not movie_reviews:
            scores[mid] = 0.0
            continue
        all_probs = []
        for i in range(0, len(movie_reviews), 8):
            batch  = movie_reviews[i:i+8]
            inputs = tokenizer(batch, return_tensors='pt',
                               truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
            all_probs.append(logits.softmax(dim=-1).detach().numpy())
        probs = np.concatenate(all_probs).mean(axis=0)
        scores[mid] = float(probs[1])
    return sorted(scores, key=scores.get, reverse=True)[:10]

def explain_recommendation(tmdb_id, min_weight=0.03):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['negative', 'positive'])
    movie_reviews = reviews_df[reviews_df.tmdbId == tmdb_id]['review'].tolist()
    if not movie_reviews:
        return 'highly rated by similar users'
    review = clean_review(movie_reviews[0])
    def predict_fn(texts):
        all_probs = []
        for i in range(0, len(texts), 8):
            inputs = tokenizer(texts[i:i+8], return_tensors='pt',
                               truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
            all_probs.append(logits.softmax(dim=-1).detach().numpy())
        return np.concatenate(all_probs, axis=0)
    exp = explainer.explain_instance(review, predict_fn,
                                     num_features=10, num_samples=300)
    keywords = [w for w, wt in exp.as_list()
                if wt > min_weight and w.lower() not in CUSTOM_STOPWORDS
                and len(w) > 3 and w.isalpha()]
    return ', '.join(keywords[:5]) if keywords else 'highly rated by similar users'

def is_known_user(user_id):
    try:
        trainset.to_inner_uid(user_id)
        return True
    except ValueError:
        return False

def get_unrated_movies(user_id):
    rated = set(ratings[ratings.userId == user_id]['movieId'].tolist())
    return list(set(merged['movieId'].tolist()) - rated)

def get_top_rated(user_id, n=20):
    return ratings[ratings.userId == user_id].nlargest(n, 'rating')['movieId'].tolist()

def recommend_cold_start(genre_preferences, n=10):
    query_vec  = tfidf.transform([' '.join(genre_preferences)])
    sims       = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx    = sims.argsort()[::-1][:300]
    candidates = merged.iloc[top_idx].copy()

    filtered = candidates[(candidates['vote_average'] >= 7.0) &
                           (candidates['vote_count']   >= 100)]

    if filtered.empty:
        filtered = candidates[candidates['vote_count'] >= 50]

    if filtered.empty:
        filtered = candidates

    return filtered.nlargest(n, 'vote_average')[['title','genres','vote_average']]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 CineIQ")
st.caption("Personalized movie recommendations powered by SVD + TF-IDF + DistilBERT")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Find your next movie")
    user_id = st.number_input("Enter User ID (leave 0 for new user)",
                               min_value=0, step=1, value=0)
    genres  = st.multiselect("Genre preferences (used for new users or to refine)",
                              ['Action','Adventure','Animation','Comedy','Crime',
                               'Documentary','Drama','Family','Fantasy','History',
                               'Horror','Music','Mystery','Romance','Science Fiction',
                               'Thriller','War','Western'])
    run = st.button("🎯 Get Recommendations", use_container_width=True)

with col2:
    if run:
        if user_id > 0 and is_known_user(user_id):
            with st.spinner("Running personalized pipeline..."):
                unrated    = get_unrated_movies(user_id)
                svd_s      = svd_scores(user_id, unrated)
                tfidf_s    = tfidf_scores(user_id, unrated, get_top_rated(user_id))
                top50      = ensemble_scores(svd_s, tfidf_s)
                top50_tmdb = merged[merged['movieId'].isin(top50)]['tmdbId'].tolist()
                top10_tmdb = rerank_with_distilbert(top50_tmdb)
                recs       = merged[merged['tmdbId'].isin(top10_tmdb)][
                                ['title','genres','vote_average','tmdbId']].copy()

            st.success(f"Top 10 picks for User {user_id}")
            for _, row in recs.iterrows():
                with st.container():
                    st.markdown(f"### {row['title']}")
                    c1, c2 = st.columns([3,1])
                    c1.caption(f"🎭 {row['genres']}")
                    c2.metric("Rating", f"{row['vote_average']:.1f}")
                    with st.spinner("Generating reason..."):
                        reason = explain_recommendation(int(row['tmdbId']))
                    st.info(f"💡 {reason}")
                    st.divider()

        elif user_id > 0 and not is_known_user(user_id):
            st.warning(f"User ID {user_id} not found. Showing recommendations based on genres or top-rated movies instead.")
            if genres:
                with st.spinner("Finding genre matches..."):
                    recs = recommend_cold_start(genres)
                st.success("Top picks based on your genre preferences")
                st.dataframe(recs, use_container_width=True)
            else:
                with st.spinner("Loading top-rated movies..."):
                    recs = merged[merged['vote_count'] >= 100].nlargest(10, 'vote_average')[
                                ['title','genres','vote_average']]
                st.success("Top rated movies")
                st.dataframe(recs, use_container_width=True)

        elif genres:
            with st.spinner("Finding genre matches..."):
                recs = recommend_cold_start(genres)
            st.success("Top picks based on your genre preferences")
            st.dataframe(recs, use_container_width=True)

        else:
            # user_id == 0 and no genres selected → show overall top-rated
            with st.spinner("Loading top-rated movies..."):
                recs = merged[merged['vote_count'] >= 100].nlargest(10, 'vote_average')[
                            ['title','genres','vote_average']]
            st.success("Top rated movies")
            st.dataframe(recs, use_container_width=True)


