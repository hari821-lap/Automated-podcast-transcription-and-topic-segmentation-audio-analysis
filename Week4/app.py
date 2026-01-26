import json
import streamlit as st

# -------- FILE PATHS --------

FULL_TRANSCRIPT_FILE = r"C:\Users\hari\Desktop\podcast\transcripts\full_transcript.txt"
SEGMENTS_FILE = r"C:\Users\hari\Desktop\podcast\week3_outputs\topic_segments.json"

# -------- LOAD FULL TRANSCRIPT --------

with open(FULL_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    full_transcript = f.read()

# -------- LOAD SEGMENTS (CASE 2 FIX) --------

with open(SEGMENTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]   # 🔥 IMPORTANT FIX

# -------- STREAMLIT UI --------

st.set_page_config(page_title="Podcast Navigator", layout="wide")

st.title("🎧 Podcast Transcript Navigation & Segment Jumping")

st.markdown("""
Browse the **full transcript** or instantly jump to **any topic segment**.
""")

# -------- SIDEBAR --------

st.sidebar.header("📌 Topics")

topic_list = ["Full Transcript"] + [
    f"{i+1}. {seg['topic']}" for i, seg in enumerate(segments)
]

choice = st.sidebar.radio("Choose View", topic_list)

# -------- DISPLAY --------

if choice == "Full Transcript":

    st.subheader("📜 Full Transcript")
    st.text_area("Complete Transcript", full_transcript, height=550)

else:
    idx = topic_list.index(choice) - 1
    seg = segments[idx]

    st.subheader(f"📌 {seg['topic']}")

    st.markdown(f"**Summary:** {seg['summary']}")
    st.markdown(f"**Keywords:** {', '.join(seg['keywords'])}")

    st.markdown("### 📜 Transcript Segment")

    st.text_area("Segment Text", seg['text'], height=450)

    st.info("⬅ Select another topic from the sidebar")
