# import os
# import json
# import nltk
# import numpy as np
# import langdetect
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from dotenv import load_dotenv
# from groq import Groq
# import soundfile as sf

# # --- ENV SETUP ---
# load_dotenv()
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # --- CONFIG ---
# nltk.download("punkt")
# TRANSCRIPT_FILE = "C:\\Users\\hari\\Desktop\\podcast\\transcripts\\full_transcript.txt"
# AUDIO_PATH = "C:\\Users\\hari\\Desktop\\podcast\\dataset\\raw_data\\NPR2877513564.mp3"
# OUTPUT_DIR = "C:\\Users\\hari\\Desktop\\podcast\\week3_outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# MODEL = "llama-3.1-8b-instant"
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # --- HELPERS ---
# def read_transcript(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read().strip()

# def detect_language(text):
#     try:
#         return langdetect.detect(text)
#     except:
#         return "unknown"

# def detect_duration_minutes(path):
#     try:
#         with sf.SoundFile(path) as f:
#             seconds = len(f) / f.samplerate
#             return round(seconds / 60, 2)
#     except:
#         return None

# # --- SAFE LLM CALL ---
# def call_llm(prompt):
#     max_len = 4000
#     chunks, buffer = [], ""
#     for line in prompt.split("\n"):
#         if len(buffer) + len(line) > max_len:
#             chunks.append(buffer)
#             buffer = ""
#         buffer += line + "\n"
#     if buffer:
#         chunks.append(buffer)

#     outputs = []
#     for c in chunks:
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=[{"role": "user", "content": c}],
#             temperature=0.2
#         )
#         outputs.append(response.choices[0].message.content.strip())
#     return "\n".join(outputs)

# # --- TITLE GENERATOR ---
# def generate_title(text):
#     safe_text = text[:3000]
#     prompt = "Generate a short meaningful title (4–6 words):\n" + safe_text
#     return call_llm(prompt)

# # --- SENTENCE SPLITTING ---
# def split_sentences(text):
#     return nltk.sent_tokenize(text)

# # --- ALGORITHM 1: SIMILARITY DROP ---
# def segment_algo1(sentences, threshold=0.55):
#     embeddings = embedder.encode(sentences)
#     segments, current = [], [sentences[0]]
#     for i in range(1, len(sentences)):
#         sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
#         if sim < threshold:
#             segments.append(" ".join(current))
#             current = [sentences[i]]
#         else:
#             current.append(sentences[i])
#     segments.append(" ".join(current))
#     return segments

# # --- ALGORITHM 2: KMEANS ---
# def segment_algo2(sentences, k=5):
#     if len(sentences) < k:
#         k = len(sentences)
#     embeddings = embedder.encode(sentences)
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(embeddings)
#     segments, current = [], [sentences[0]]
#     current_label = labels[0]
#     for i in range(1, len(sentences)):
#         if labels[i] != current_label:
#             segments.append(" ".join(current))
#             current = [sentences[i]]
#             current_label = labels[i]
#         else:
#             current.append(sentences[i])
#     segments.append(" ".join(current))
#     return segments

# # --- ALGORITHM 3: LLM ---
# def segment_algo3(text):
#     safe_text = text[:4000]
#     prompt = (
#         "Split the following transcript into topic-based segments.\n"
#         "Return ONLY a JSON list of text segments.\n\n" + safe_text
#     )
#     try:
#         return json.loads(call_llm(prompt))
#     except:
#         return [safe_text]

# # --- KEYWORDS ---
# def extract_keywords(text, top_n=6):
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf = vectorizer.fit_transform([text])
#     scores = tfidf.toarray()[0]
#     words = vectorizer.get_feature_names_out()
#     idx = np.argsort(scores)[-top_n:]
#     return [words[i] for i in idx]

# # --- SUMMARIES ---
# def summarize_segment(text):
#     prompt = "Summarize the following segment in 2 sentences:\n" + text
#     return call_llm(prompt)

# # --- SAVE SEGMENTS ---
# def save_segments(segments, summaries, output_file):
#     with open(output_file, "w", encoding="utf-8") as f:
#         for i, seg in enumerate(segments):
#             f.write(f"Segment {i+1}\n")
#             f.write("Text:\n")
#             f.write(seg.strip() + "\n\n")
#             f.write("Summary:\n")
#             f.write(summaries[i] + "\n")
#             f.write("-" * 50 + "\n\n")

# # --- MAIN PIPELINE ---
# print("Reading transcript...")
# transcript = read_transcript(TRANSCRIPT_FILE)

# print("Generating overall title...")
# overall_title = generate_title(transcript)

# print("Sentence tokenization...")
# sentences = split_sentences(transcript)

# print("Running Algorithm 1...")
# algo1_segments = segment_algo1(sentences)

# print("Running Algorithm 2...")
# algo2_segments = segment_algo2(sentences)

# print("Running Algorithm 3 (LLM)...")
# final_segments = segment_algo3(transcript)

# print("Generating summaries...")
# summaries = [summarize_segment(seg) for seg in final_segments]

# print("Extracting keywords...")
# keywords = {f"Segment {i+1}": extract_keywords(seg) for i, seg in enumerate(final_segments)}

# print("Saving final output...")
# save_segments(final_segments, summaries, os.path.join(OUTPUT_DIR, "segmented_output.txt"))

# with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
#     json.dump({
#         "title": overall_title,
#         "language": detect_language(transcript),
#         "duration_minutes": detect_duration_minutes(AUDIO_PATH),
#         "algorithm_1_segments": len(algo1_segments),
#         "algorithm_2_segments": len(algo2_segments),
#         "final_segments": len(final_segments)
#     }, f, indent=4)

# with open(os.path.join(OUTPUT_DIR, "summaries.json"), "w", encoding="utf-8") as f:
#     json.dump(summaries, f, indent=4)

# with open(os.path.join(OUTPUT_DIR, "keywords.json"), "w", encoding="utf-8") as f:
#     json.dump(keywords, f, indent=4)

# print("Week 3 pipeline completed successfully!")

# import os
# import json
# import re
# import nltk
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from groq import Groq

# BASE_DIR = "outputs"
# MODEL_NAME = "llama-3.1-8b-instant"
# CHUNK_SIZE = 1200
# TOTAL_DURATION = 3600  # seconds (1 hour estimate)

# nltk.download("punkt", quiet=True)

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # ---------------- HELPERS ----------------

# def extract_json(text):
#     match = re.search(r"\[\s*{.*?}\s*\]", text, re.S)
#     return match.group() if match else None

# def chunk_text(text, size):
#     return [text[i:i+size] for i in range(0, len(text), size)]

# # ---------------- TOPIC SEGMENTATION ----------------

# def topic_segmentation(text):
#     segments = []

#     for chunk in chunk_text(text, CHUNK_SIZE):
#         prompt = f"""
# Split transcript into topic segments.
# Return JSON only:

# [{{"topic":"...", "text":"..."}}]

# Transcript:
# {chunk}
# """
#         res = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1
#         )

#         raw = res.choices[0].message.content.strip()
#         js = extract_json(raw)

#         if js:
#             segments.extend(json.loads(js))
#         else:
#             segments.append({"topic": "General", "text": chunk})

#     return segments

# # ---------------- KEYWORDS ----------------

# def extract_keywords(segments):
#     texts = [s["text"] for s in segments]
#     vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
#     tfidf = vectorizer.fit_transform(texts)
#     words = vectorizer.get_feature_names_out()

#     for i, seg in enumerate(segments):
#         scores = tfidf[i].toarray()[0]
#         top_idx = scores.argsort()[-6:][::-1]
#         seg["keywords"] = [words[j] for j in top_idx]

#     return segments

# # ---------------- SUMMARIES ----------------

# def generate_summaries(segments):
#     for seg in segments:
#         seg["summary"] = " ".join(sent_tokenize(seg["text"])[:2])
#     return segments

# # ---------------- TIMESTAMPS (FIXED) ----------------

# def assign_timestamps(segments, duration=TOTAL_DURATION):
#     total = len(segments)
#     for i, seg in enumerate(segments):
#         start = int((i / total) * duration)
#         end = int(((i + 1) / total) * duration)
#         seg["timestamp"] = {
#             "start": start,
#             "end": end
#         }
#     return segments

# # ---------------- MAIN ----------------

# def process_all():
#     for podcast in os.listdir(BASE_DIR):
#         transcript_path = os.path.join(
#             BASE_DIR, podcast, "transcript", "full_transcript.txt"
#         )

#         if not os.path.exists(transcript_path):
#             continue

#         print("Processing:", podcast)

#         week3_dir = os.path.join(BASE_DIR, podcast, "week3")
#         os.makedirs(week3_dir, exist_ok=True)

#         text = open(transcript_path, encoding="utf-8").read()

#         segments = topic_segmentation(text)
#         segments = extract_keywords(segments)
#         segments = generate_summaries(segments)
#         segments = assign_timestamps(segments)  # ✅ FIX

#         out = os.path.join(week3_dir, "topic_segments.json")
#         json.dump({"segments": segments}, open(out, "w", encoding="utf-8"), indent=4)

#         print("Saved →", out)

# process_all()


import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

TRANSCRIPT_DIR = "transcripts"
OUTPUT_BASE = "outputs"

MODEL_NAME = "llama-3.1-8b-instant"
SEGMENT_DURATION = 180  # 3 minutes

nltk.download("punkt", quiet=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- HELPERS ----------------

def seconds_to_mmss(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def extract_json(text):
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.S)
    return match.group() if match else None

# ---------------- LOAD WORD TIMINGS ----------------

def load_word_timings(base_name):
    words = []

    files = sorted([
        f for f in os.listdir(TRANSCRIPT_DIR)
        if f.startswith(base_name) and f.endswith(".json")
    ])

    for f in files:
        with open(os.path.join(TRANSCRIPT_DIR, f), encoding="utf-8") as jf:
            data = json.load(jf)
            for w in data.get("result", []):
                words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"]
                })

    return words

# ---------------- BUILD 3-MINUTE SEGMENTS ----------------

def build_time_segments(words):

    segments = []
    if not words:
        return segments

    current = []
    seg_start = words[0]["start"]

    for w in words:
        current.append(w["word"])

        if w["end"] - seg_start >= SEGMENT_DURATION:
            segments.append({
                "text": " ".join(current),
                "start": seg_start,
                "end": w["end"]
            })
            current = []
            seg_start = w["end"]

    if current:
        segments.append({
            "text": " ".join(current),
            "start": seg_start,
            "end": words[-1]["end"]
        })

    return segments

# ---------------- TOPIC SEGMENTATION ----------------

def topic_segmentation(time_segments):

    final_segments = []

    for seg in time_segments:
        prompt = f"""
Find the main topic of this podcast segment.
Return JSON only:

[{{"topic":"...", "text":"..."}}]

Transcript:
{seg["text"]}
"""

        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        raw = res.choices[0].message.content.strip()
        js = extract_json(raw)

        if js:
            data = json.loads(js)[0]
            data["start"] = seg["start"]
            data["end"] = seg["end"]
            final_segments.append(data)
        else:
            final_segments.append({
                "topic": "General",
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"]
            })

    return final_segments

# ---------------- KEYWORDS ----------------

def extract_keywords(segments):
    texts = [s["text"] for s in segments]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    for i, seg in enumerate(segments):
        scores = tfidf[i].toarray()[0]
        top_idx = scores.argsort()[-6:][::-1]
        seg["keywords"] = [words[j] for j in top_idx]

    return segments

# ---------------- SUMMARY ----------------

def generate_summaries(segments):
    for seg in segments:
        seg["summary"] = " ".join(sent_tokenize(seg["text"])[:2])
    return segments

# ---------------- FORMAT TIMESTAMPS ----------------

def format_timestamps(segments):
    for seg in segments:
        seg["timestamp"] = {
            "start": seconds_to_mmss(seg["start"]),
            "end": seconds_to_mmss(seg["end"])
        }
        del seg["start"]
        del seg["end"]
    return segments

# ---------------- MAIN ----------------

def process_all():

    transcripts = [
        f for f in os.listdir(TRANSCRIPT_DIR)
        if f.endswith("_full_transcript.txt")
    ]

    for file in transcripts:

        base_name = file.replace("_full_transcript.txt", "")
        print("Processing:", base_name)

        words = load_word_timings(base_name)
        time_segments = build_time_segments(words)

        segments = topic_segmentation(time_segments)
        segments = extract_keywords(segments)
        segments = generate_summaries(segments)
        segments = format_timestamps(segments)

        podcast_dir = os.path.join(OUTPUT_BASE, base_name, "week3")
        os.makedirs(podcast_dir, exist_ok=True)

        out = os.path.join(podcast_dir, "topic_segments.json")

        with open(out, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, indent=4)

        print("Saved →", out)

process_all()













