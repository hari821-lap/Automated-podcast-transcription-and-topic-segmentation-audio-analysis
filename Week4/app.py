import os
import json
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

BASE_DIR = "outputs"

st.set_page_config(
    page_title="Podcast AI Navigator",
    layout="wide"
)

st.title("🎧 Podcast AI Navigator")

# ---------------- LOAD PODCASTS ----------------

if not os.path.exists(BASE_DIR):
    st.error("❌ Outputs folder not found")
    st.stop()

podcasts = [
    p for p in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, p))
]

selected_podcast = st.selectbox("🎙 Select Podcast", podcasts)

# ---------------- LOAD DATA ----------------

segments_path = os.path.join(
    BASE_DIR, selected_podcast, "week3", "topic_segments.json"
)

transcript_path = os.path.join(
    BASE_DIR, selected_podcast, "transcript", "full_transcript.txt"
)

with open(segments_path, encoding="utf-8") as f:
    segments = json.load(f)["segments"]

with open(transcript_path, encoding="utf-8") as f:
    full_text = f.read()

# ---------------- SESSION STATE ----------------

if "selected_index" not in st.session_state:
    st.session_state.selected_index = 0

# ---------------- COLORS ----------------

COLORS = [
    "#4CAF50", "#2196F3", "#FF9800",
    "#9C27B0", "#F44336", "#00BCD4",
    "#795548", "#3F51B5"
]

def color(i):
    return COLORS[i % len(COLORS)]

# ---------------- SIDEBAR ----------------

st.sidebar.header("📚 Topics")

topics = [f"{i+1}. {s['topic']}" for i, s in enumerate(segments)]
choice = st.sidebar.radio("Navigate", topics)

st.session_state.selected_index = int(choice.split(".")[0]) - 1

# ---------------- TIMELINE ----------------

st.subheader("🕒 Podcast Timeline")

total_segments = len(segments)

selected = st.slider(
    "Move across podcast",
    min_value=1,
    max_value=total_segments,
    value=st.session_state.selected_index + 1
)

st.session_state.selected_index = selected - 1

# ---- VISUAL SEGMENT BAR ----

bar_html = ""
for i in range(total_segments):
    bar_html += f"""
    <div style="
        flex:1;
        height:10px;
        background:{color(i)};
        margin-right:2px;
        border-radius:4px;">
    </div>
    """

st.markdown(
    f"""
    <div style="display:flex; width:100%; margin-top:-10px;">
        {bar_html}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- DISPLAY CONTENT ----------------

seg = segments[st.session_state.selected_index]

st.header(f"🎯 Topic {st.session_state.selected_index + 1}: {seg['topic']}")

left, right = st.columns([1, 1.6])

with left:
    st.subheader("📝 Summary")
    st.write(seg.get("summary", "Not available"))

    st.subheader("🔑 Keywords")
    st.write(", ".join(seg.get("keywords", [])))

    ts = seg.get("timestamp", {})
    st.subheader("⏱ Timestamp")
    st.write(f"{ts.get('start', '?')}s → {ts.get('end', '?')}s")

    # ---- WORD CLOUD ----
    st.subheader("☁ Word Cloud")

    wc_text = " ".join(seg.get("keywords", [])) or seg.get("text", "")
    wc = WordCloud(
        width=400,
        height=300,
        background_color="white"
    ).generate(wc_text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

with right:
    st.subheader("📄 Segment Transcript")
    st.text_area(
        "Segment Text",
        seg.get("text", ""),
        height=420
    )

# ---------------- FULL TRANSCRIPT ----------------

with st.expander("📜 View Full Transcript"):
    st.text_area("Transcript", full_text, height=400)
