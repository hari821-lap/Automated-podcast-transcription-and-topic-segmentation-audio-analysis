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

import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq

# ---------------- CONFIG ----------------
TRANSCRIPT_FILE = r"C:\Users\hari\Desktop\podcast\transcripts\full_transcript.txt"
OUTPUT_DIR = r"C:\Users\hari\Desktop\podcast\week3_outputs"

CHUNK_SIZE = 1200
MODEL_NAME = "llama-3.1-8b-instant"

# API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Run: setx GROQ_API_KEY \"your_key\"")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- HELPERS ----------------

def extract_json(text):
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.S)
    if match:
        return match.group()
    return None

# ---------------- CORE PIPELINE ----------------

def load_transcript(path):
    print("Loading transcript...")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, size):
    print("Splitting into small chunks...")
    return [text[i:i+size] for i in range(0, len(text), size)]

def groq_topic_segmentation(chunks):
    print("Running LLM topic segmentation using GROQ...")
    client = Groq(api_key=GROQ_API_KEY)
    all_segments = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}")

        prompt = f"""
You are an NLP system.

Split the transcript into logical topic segments.

Return ONLY valid JSON.

FORMAT:
[
  {{"topic":"...", "text":"..."}}
]

Transcript:
{chunk}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            raw = response.choices[0].message.content.strip()
            json_text = extract_json(raw)

            if not json_text:
                print("⚠ JSON parse failed → fallback used")
                all_segments.append({"topic": "General Discussion", "text": chunk})
                continue

            segments = json.loads(json_text)
            all_segments.extend(segments)

        except Exception as e:
            print("⚠ LLM Error:", e)
            all_segments.append({"topic": "General Discussion", "text": chunk})

    return all_segments

def extract_keywords(segments, top_n=6):
    print("Extracting keywords (TF-IDF)...")
    texts = [s["text"] for s in segments]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()

    for i, seg in enumerate(segments):
        scores = tfidf[i].toarray()[0]
        top_idx = scores.argsort()[-top_n:][::-1]
        seg["keywords"] = [words[j] for j in top_idx]

    return segments

def generate_summaries(segments):
    print("Generating short summaries...")
    for seg in segments:
        sents = sent_tokenize(seg["text"])
        seg["summary"] = " ".join(sents[:2])
    return segments

def save_outputs(segments):
    json_path = os.path.join(OUTPUT_DIR, "topic_segments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, indent=4, ensure_ascii=False)

    print("\n✅ Output saved:", json_path)

def preview(segments, n=5):
    print("\n🔹 Preview:\n")
    for i, seg in enumerate(segments[:n], 1):
        print(f"{i}. {seg['topic']}")
        print("   Keywords:", ", ".join(seg['keywords']))
        print("   Summary:", seg['summary'])
        print("-"*80)

# ---------------- MAIN ----------------

def main():
    nltk.download("punkt", quiet=True)

    transcript = load_transcript(TRANSCRIPT_FILE)
    chunks = chunk_text(transcript, CHUNK_SIZE)

    segments = groq_topic_segmentation(chunks)
    segments = extract_keywords(segments)
    segments = generate_summaries(segments)

    save_outputs(segments)
    preview(segments)

if __name__ == "__main__":
    main()










