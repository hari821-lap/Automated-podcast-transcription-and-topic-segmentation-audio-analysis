import os
import json
import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = "outputs"
AUDIO_DIR = r"C:\Users\hari\Desktop\podcast\dataset\raw_data"

st.set_page_config(
    page_title="🎧 Podcast Visualization (Week 5)",
    layout="wide"
)

# ---------------- HELPERS ----------------
def sentiment_info(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive", "#4CAF50"
    elif polarity < -0.1:
        return "Negative", "#F44336"
    else:
        return "Neutral", "#BDBDBD"

# convert seconds → mm:ss
def sec_to_time(sec):
    sec = int(sec)
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

# convert mm:ss OR seconds → seconds
def parse_time(value):
    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        value = value.strip()
        if ":" in value:
            m, s = value.split(":")
            return int(m) * 60 + int(s)
        else:
            return int(float(value))

    return 0

def get_audio_file():
    for f in os.listdir(AUDIO_DIR):
        if f.endswith((".mp3", ".wav", ".m4a")):
            return os.path.join(AUDIO_DIR, f)
    return None

# ---------------- TITLE ----------------
st.title("🎧 Podcast Timeline & Topic Explorer")

# ---------------- LOAD PODCASTS ----------------
if not os.path.exists(BASE_DIR):
    st.error("Outputs folder not found")
    st.stop()

podcasts = [
    p for p in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, p))
]

selected_podcast = st.selectbox("🎙 Select Podcast", podcasts)

# ---------------- LOAD SEGMENTS ----------------
segments_path = os.path.join(
    BASE_DIR, selected_podcast, "week3", "topic_segments.json"
)

if not os.path.exists(segments_path):
    st.error("topic_segments.json not found")
    st.stop()

with open(segments_path, encoding="utf-8") as f:
    segments = json.load(f)["segments"]

# ✅ FIX TIMESTAMPS HERE
for s in segments:
    s["timestamp"]["start"] = parse_time(s["timestamp"]["start"])
    s["timestamp"]["end"] = parse_time(s["timestamp"]["end"])

# add sentiment
for s in segments:
    label, color = sentiment_info(s["text"])
    s["sentiment"] = label
    s["color"] = color

# ---------------- SESSION STATE ----------------
if "index" not in st.session_state:
    st.session_state.index = 0

# ---------------- SIDEBAR ----------------
st.sidebar.header("📚 Topic Navigation")

topic_list = [f"{i+1}. {s['topic']}" for i, s in enumerate(segments)]
selected_topic = st.sidebar.radio(
    "Jump to topic",
    topic_list,
    index=st.session_state.index
)

st.session_state.index = topic_list.index(selected_topic)
seg = segments[st.session_state.index]

# ---------------- TIMELINE ----------------
st.subheader("🕒 Podcast Timeline")

timeline_html = ""
for i, s in enumerate(segments):
    start = sec_to_time(s["timestamp"]["start"])
    end = sec_to_time(s["timestamp"]["end"])
    border = "2px solid black" if i == st.session_state.index else "none"

    timeline_html += f"""
    <div title="{s['topic']} ({start} → {end})"
         style="
            flex:1;
            height:6px;
            background:{s['color']};
            border:{border};
            border-radius:3px;">
    </div>
    """

st.markdown(
    f"""
    <div style="
        display:flex;
        gap:2px;
        width:100%;
        margin-bottom:6px;">
        {timeline_html}
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("🟢 Positive | 🔴 Negative | ⚪ Neutral")

# ---------------- SLIDER ----------------
st.slider(
    "Navigate timeline",
    1,
    len(segments),
    st.session_state.index + 1,
    key="timeline_slider",
    on_change=lambda: st.session_state.update(
        index=st.session_state.timeline_slider - 1
    )
)

# ---------------- AUDIO ----------------
st.subheader("🔊 Podcast Audio (Synced)")

audio_file = get_audio_file()

if audio_file:
    st.audio(audio_file, start_time=seg["timestamp"]["start"])
else:
    st.warning("⚠ No audio file found in raw_data folder")

# ---------------- DETAILS ----------------
st.markdown("---")
st.header(f"🎯 {seg['topic']}")

col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("📝 Summary")
    st.write(seg["summary"])

    st.subheader("🙂 Sentiment")
    st.write(seg["sentiment"])

    st.subheader("🔑 Keywords")
    st.write(", ".join(seg["keywords"]))

with col2:
    st.subheader("📄 Transcript")
    st.text_area(
        "Segment Text",
        seg["text"],
        height=300
    )

# ---------------- WORD CLOUD ----------------
st.subheader("☁ Keyword Cloud")

wc_text = " ".join(seg["keywords"])
wc = WordCloud(
    width=600,
    height=300,
    background_color="white"
).generate(wc_text)

plt.figure(figsize=(6, 3))
plt.imshow(wc)
plt.axis("off")
st.pyplot(plt)
