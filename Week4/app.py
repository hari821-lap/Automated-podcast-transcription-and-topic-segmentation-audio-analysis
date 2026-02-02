import os
import json
import streamlit as st

# ---------------- CONFIG ----------------

BASE_DIR = "outputs"

st.set_page_config(
    page_title="Podcast Transcript Navigator",
    layout="wide"
)

st.title("🎧 Podcast Transcript Navigator")

# ---------------- LOAD PODCASTS ----------------

if not os.path.exists(BASE_DIR):
    st.error("❌ Outputs directory not found")
    st.stop()

podcasts = [
    p for p in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, p))
]

if not podcasts:
    st.warning("⚠ No podcasts found in outputs folder")
    st.stop()

selected_podcast = st.selectbox("🎙 Select Podcast", podcasts)

# ---------------- LOAD FILES ----------------

transcript_path = os.path.join(
    BASE_DIR, selected_podcast, "transcript", "full_transcript.txt"
)

segments_path = os.path.join(
    BASE_DIR, selected_podcast, "week3", "topic_segments.json"
)

if not os.path.exists(transcript_path):
    st.error("❌ full_transcript.txt not found")
    st.stop()

if not os.path.exists(segments_path):
    st.error("❌ topic_segments.json not found")
    st.stop()

with open(transcript_path, encoding="utf-8") as f:
    full_text = f.read()

with open(segments_path, encoding="utf-8") as f:
    segments_data = json.load(f)

segments = segments_data.get("segments", [])

if not segments:
    st.warning("⚠ No topic segments available")

# ---------------- SIDEBAR NAVIGATION ----------------

st.sidebar.header("📚 Navigate Topics")

nav_items = ["📜 Full Transcript"]
nav_items += [f"{i+1}. {seg['topic']}" for i, seg in enumerate(segments)]

choice = st.sidebar.radio(
    "Go to",
    nav_items,
    key="navigation_radio"
)

# ---------------- DISPLAY ----------------

st.markdown("---")

# ===== FULL TRANSCRIPT VIEW =====
if choice == "📜 Full Transcript":
    st.header("📜 Full Transcript")
    st.text_area(
        "Transcript",
        full_text,
        height=550,
        key="full_transcript_area"
    )

# ===== TOPIC VIEW =====
else:
    index = int(choice.split(".")[0]) - 1
    seg = segments[index]

    st.header(f"🎯 Topic: {seg.get('topic', 'N/A')}")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("📝 Summary")
        st.write(seg.get("summary", "Not available"))

        st.subheader("🔑 Keywords")
        keywords = seg.get("keywords", [])
        st.write(", ".join(keywords) if keywords else "Not available")

    with col2:
        st.subheader("📄 Segment Text")
        st.text_area(
            "Segment Content",
            seg.get("text", ""),
            height=350,
            key=f"segment_text_{index}"
        )
