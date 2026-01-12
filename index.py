import os
import json
import librosa
import soundfile as sf
from vosk import Model, KaldiRecognizer
import numpy as np

RAW_DIR = "dataset/raw_data/"
WAV_DIR = "preprocessed_audio/wav16k/"
NORM_DIR = "preprocessed_audio/normalized/"
CHUNK_DIR = "preprocessed_audio/chunks/"
TRANSCRIPT_DIR = "transcripts/"

for d in [WAV_DIR, NORM_DIR, CHUNK_DIR, TRANSCRIPT_DIR]:
    os.makedirs(d, exist_ok=True)

print("Loading Vosk model...")
model = Model("models/vosk-model-small-en-us-0.15")


def convert_to_wav16k(filepath):
    print("Converting:", filepath)
    try:
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        out = os.path.join(WAV_DIR, os.path.basename(filepath).split(".")[0] + ".wav")
        sf.write(out, audio, 16000)
        return out
    except Exception as e:
        print("Conversion error:", e)
        return None


def normalize_audio(filepath):
    print("Normalizing:", os.path.basename(filepath))
    try:
        audio, sr = librosa.load(filepath, sr=16000)
        max_val = np.max(np.abs(audio)) + 1e-9
        audio = audio / max_val
        out = os.path.join(NORM_DIR, os.path.basename(filepath))
        sf.write(out, audio, sr)
        return out
    except:
        return None


def chunk_audio(filepath, chunk_sec=25):
    print("Chunking:", os.path.basename(filepath))
    audio, sr = librosa.load(filepath, sr=16000)

    chunk_samples = chunk_sec * sr
    total = len(audio)

    base = os.path.splitext(os.path.basename(filepath))[0]
    chunks = []

    start = 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunk = audio[start:end]
        out_path = os.path.join(CHUNK_DIR, f"{base}_{start//sr}.wav")
        sf.write(out_path, chunk, sr)
        chunks.append(out_path)
        start += chunk_samples

    print("Total chunks:", len(chunks))
    return chunks


def transcribe(chunk_path):
    print("Transcribing:", os.path.basename(chunk_path))

    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    audio, sr = sf.read(chunk_path)

    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    step = 4000
    for i in range(0, len(audio), step):
        rec.AcceptWaveform(audio[i:i + step].tobytes())

    result = json.loads(rec.FinalResult())

    out_json = chunk_path.replace("preprocessed_audio/chunks/", "transcripts/").replace(".wav", ".json")

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved:", out_json)


#  MERGE ALL JSON FILES 
def merge_transcripts():
    print("\nMerging all transcripts into one file...")

    txt_output = os.path.join(TRANSCRIPT_DIR, "full_transcript.txt")

    with open(txt_output, "w", encoding="utf-8") as outfile:
        # Process JSON in sorted order to maintain time sequence
        files = sorted([f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".json")])

        for f in files:
            path = os.path.join(TRANSCRIPT_DIR, f)
            with open(path, "r") as jf:
                data = json.load(jf)

                if "text" in data and data["text"].strip():
                    outfile.write(data["text"] + "\n\n")

    print("Merged transcript saved at:", txt_output)

# Main pipeline

def run():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith((".mp3", ".wav"))]

    for f in files:
        print("\nProcessing:", f)
        raw_path = os.path.join(RAW_DIR, f)

        wav = convert_to_wav16k(raw_path)
        if not wav:
            continue

        norm = normalize_audio(wav)
        if not norm:
            continue

        chunks = chunk_audio(norm)

        for c in chunks:
            transcribe(c)

    # Merge final transcripts
    merge_transcripts()

    print("\nAll processing completed.")


run()
