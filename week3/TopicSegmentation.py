import os
import json
import nltk
import numpy as np
import librosa
import langdetect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from groq import Groq

# ENV SETUP

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CONFIG

nltk.download("punkt")

TRANSCRIPT_FILE = "full_transcript.txt"
AUDIO_PATH = "dataset/raw_data/NPR2877513564.mp3"
OUTPUT_DIR = "week3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "llama-3.1-8b-instant"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# HELPERS

def read_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

def detect_duration(path):
    try:
        audio, sr = librosa.load(path)
        return round(len(audio) / sr, 2)
    except:
        return None


# SAFE LLM CALL (CHUNKING)

def call_llm(prompt):
    max_len = 4000
    chunks, buffer = [], ""

    for line in prompt.split("\n"):
        if len(buffer) + len(line) > max_len:
            chunks.append(buffer)
            buffer = ""
        buffer += line + "\n"

    if buffer:
        chunks.append(buffer)

    outputs = []
    for c in chunks:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": c}],
            temperature=0.2
        )
        outputs.append(response.choices[0].message.content.strip())

    return "\n".join(outputs)


# TITLE GENERATOR (TOKEN SAFE)

def generate_title(text):
    safe_text = text[:3000]
    prompt = "Generate a short meaningful title (4–6 words):\n" + safe_text
    return call_llm(prompt)


# STEP 2: SENTENCE SPLITTING

def split_sentences(text):
    return nltk.sent_tokenize(text)


# STEP 3: ALGORITHM 1 (BASELINE – SIMILARITY DROP)

def segment_algo1(sentences, threshold=0.55):
    embeddings = embedder.encode(sentences)
    segments, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if sim < threshold:
            segments.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    segments.append(" ".join(current))
    return segments


# STEP 4: ALGORITHM 2 (EMBEDDING + KMEANS)

def segment_algo2(sentences, k=5):
    if len(sentences) < k:
        k = len(sentences)

    embeddings = embedder.encode(sentences)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    segments, current = [], [sentences[0]]
    current_label = labels[0]

    for i in range(1, len(sentences)):
        if labels[i] != current_label:
            segments.append(" ".join(current))
            current = [sentences[i]]
            current_label = labels[i]
        else:
            current.append(sentences[i])

    segments.append(" ".join(current))
    return segments


# STEP 5: ALGORITHM 3 (LLM BASED SEGMENTATION)

def segment_algo3(text):
    safe_text = text[:4000]
    prompt = (
        "Split the following transcript into topic-based segments.\n"
        "Return ONLY a JSON list of text segments.\n\n"
        + safe_text
    )
    try:
        return json.loads(call_llm(prompt))
    except:
        return [safe_text]

# KEYWORDS (TF-IDF)

def extract_keywords(text, top_n=6):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray()[0]
    words = vectorizer.get_feature_names_out()
    idx = np.argsort(scores)[-top_n:]
    return [words[i] for i in idx]


# SUMMARY GENERATION

def summarize_segment(text):
    prompt = "Summarize the following segment in 2 sentences:\n" + text
    return call_llm(prompt)


# SAVE FINAL OUTPUT (REQUIRED FORMAT)

def save_segments(segments, summaries, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            f.write(f"Segment {i+1}\n")
            f.write("Text:\n")
            f.write(seg.strip() + "\n\n")
            f.write("Summary:\n")
            f.write(summaries[i] + "\n")
            f.write("-" * 50 + "\n\n")


# MAIN PIPELINE

print("Reading transcript")
transcript = read_transcript(TRANSCRIPT_FILE)

print("Generating overall title")
overall_title = generate_title(transcript)

print("Sentence tokenization")
sentences = split_sentences(transcript)

print("Running Algorithm 1")
algo1_segments = segment_algo1(sentences)

print("Running Algorithm 2")
algo2_segments = segment_algo2(sentences)

print("Running Algorithm 3")
final_segments = segment_algo3(transcript)

print("Generating summaries")
summaries = [summarize_segment(seg) for seg in final_segments]

print("Saving final output")
save_segments(
    final_segments,
    summaries,
    os.path.join(OUTPUT_DIR, "segmented_output.txt")
)

metadata = {
    "title": overall_title,
    "language": detect_language(transcript),
    "duration_seconds": detect_duration(AUDIO_PATH),
    "algorithm_1_segments": len(algo1_segments),
    "algorithm_2_segments": len(algo2_segments),
    "final_segments": len(final_segments)
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("Pipeline completed successfully")
print("Segment titles, text, and summaries saved in:", OUTPUT_DIR)


