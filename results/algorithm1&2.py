import os
import json
import nltk
import numpy as np

from nltk.tokenize import sent_tokenize, TextTilingTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG & FOLDER SETUP

# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Transcript path (auto-detect)
TRANSCRIPT_PATH = os.path.join(BASE_DIR, "..", "transcripts", "full_transcript.txt")

# Results folders
RESULT_DIR = os.path.join(BASE_DIR, "..", "results")
ALGO1_DIR = os.path.join(RESULT_DIR, "algorithm1")
ALGO2_DIR = os.path.join(RESULT_DIR, "algorithm2")

# Create folders if they don't exist
for folder in [ALGO1_DIR, ALGO2_DIR]:
    os.makedirs(folder, exist_ok=True)

# NLTK downloads
nltk.download("punkt")
nltk.download("punkt_tab")

# LOAD TRANSCRIPT

def load_transcript():
    if not os.path.exists(TRANSCRIPT_PATH):
        raise FileNotFoundError(f"Transcript not found at: {TRANSCRIPT_PATH}")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ALGORITHM 1: Sentence Similarity

def algorithm1_segment(text, threshold=0.35):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer().fit(sentences)
    vectors = vectorizer.transform(sentences).toarray()

    segments = []
    seg_points = []

    for i in range(len(sentences) - 1):
        sim = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
        if sim < threshold:
            seg_points.append(i + 1)

    last = 0
    for p in seg_points:
        segments.append(" ".join(sentences[last:p]))
        last = p
    segments.append(" ".join(sentences[last:]))

    return segments

# ALGORITHM 2: TextTiling

def algorithm2_segment(text):
    try:
        tokenizer = TextTilingTokenizer()
        return tokenizer.tokenize(text)
    except:
        return [text]

# KEYWORD EXTRACTION

def extract_keywords(segments, top_n=8):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(segments)
    feature_names = vectorizer.get_feature_names_out()

    keywords = {}
    for idx, seg in enumerate(segments):
        row = tfidf_matrix[idx].toarray().flatten()
        top_idx = np.argsort(row)[-top_n:][::-1]
        keywords[f"segment_{idx+1}"] = [feature_names[i] for i in top_idx]
    return keywords

# SUMMARY CREATION

def generate_summaries(segments):
    summaries = {}
    for i, seg in enumerate(segments):
        sentences = sent_tokenize(seg)
        summary = " ".join(sentences[:2]) if len(sentences) >= 2 else seg
        summaries[f"segment_{i+1}"] = summary
    return summaries

# SAVE UTILITIES

def save_segments(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"\n\n======== SEGMENT {i} ========\n")
            f.write(seg)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# MAIN PIPELINE

def main():
    print("Loading transcript...")
    text = load_transcript()

    # --- Algorithm 1 ---
    print("\nRunning Algorithm 1 (Sentence Similarity)...")
    algo1_segments = algorithm1_segment(text)
    save_segments(os.path.join(ALGO1_DIR, "segments.txt"), algo1_segments)

    keywords1 = extract_keywords(algo1_segments)
    save_json(os.path.join(ALGO1_DIR, "keywords.json"), keywords1)

    summaries1 = generate_summaries(algo1_segments)
    save_json(os.path.join(ALGO1_DIR, "summaries.json"), summaries1)

    # --- Algorithm 2 ---
    print("\nRunning Algorithm 2 (TextTiling)...")
    algo2_segments = algorithm2_segment(text)
    save_segments(os.path.join(ALGO2_DIR, "segments.txt"), algo2_segments)

    keywords2 = extract_keywords(algo2_segments)
    save_json(os.path.join(ALGO2_DIR, "keywords.json"), keywords2)

    summaries2 = generate_summaries(algo2_segments)
    save_json(os.path.join(ALGO2_DIR, "summaries.json"), summaries2)

    print("\nDONE! All results saved in /results/algorithm1 and /results/algorithm2 folders.")


# RUN


if __name__ == "__main__":
    main()