import os
import json
import librosa
import soundfile as sf
from vosk import Model, KaldiRecognizer
import numpy as np

# FOLDERS
RAW_DIR = "dataset/raw_data/"
WAV_DIR = "preprocessed_audio/wav16k/"
NORM_DIR = "preprocessed_audio/normalized/"
CHUNK_DIR = "preprocessed_audio/chunks/"
TRANSCRIPT_DIR = "transcripts/"

# CREATE DIRECTORIES
for d in [WAV_DIR, NORM_DIR, CHUNK_DIR, TRANSCRIPT_DIR]:
    os.makedirs(d, exist_ok=True)

# LOAD MODEL
print("Loading Vosk Model...")

model = Model(r"C:\Users\hari\Desktop\podcast\models\vosk-model-small-en-us-0.15")

print("Vosk Model Loaded Successfully.\n")


# CONVERT TO WAV 16k
def convert_to_wav16k(filepath):
    print(f"Converting to 16k WAV -> {os.path.basename(filepath)}")

    try:
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        out = os.path.join(WAV_DIR, os.path.splitext(os.path.basename(filepath))[0] + ".wav")
        sf.write(out, audio, 16000)
        print("   Conversion completed.")
        return out
    except Exception as e:
        print("   Conversion error:", e)
        return None


# NORMALIZE AUDIO
def normalize_audio(filepath):
    print(f"Normalizing audio -> {os.path.basename(filepath)}")

    try:
        audio, sr = librosa.load(filepath, sr=16000)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        out = os.path.join(NORM_DIR, os.path.basename(filepath))
        sf.write(out, audio, sr)
        print("   Normalization completed.")
        return out
    except Exception as e:
        print("   Normalize error:", e)
        return None


# CHUNK AUDIO
def chunk_audio(filepath, chunk_sec=20):
    print(f"Splitting audio into {chunk_sec}-second chunks -> {os.path.basename(filepath)}")

    audio, sr = librosa.load(filepath, sr=16000)
    chunk_samples = chunk_sec * sr
    total = len(audio)

    base = os.path.splitext(os.path.basename(filepath))[0]
    chunks = []

    start = 0
    index = 1
    while start < total:
        end = min(start + chunk_samples, total)
        chunk = audio[start:end]

        out_name = f"{base}_chunk{index}.wav"
        out_path = os.path.join(CHUNK_DIR, out_name)

        sf.write(out_path, chunk, sr)
        chunks.append(out_path)

        print(f"   Created chunk {index}")

        start += chunk_samples
        index += 1

    print(f"   Total chunks created: {len(chunks)}\n")
    return chunks


# TRANSCRIBE AUDIO
def transcribe(chunk_path):
    print(f"Transcribing -> {os.path.basename(chunk_path)}")

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

    print(f"   Saved transcript: {os.path.basename(out_json)}\n")


# MERGE ALL TRANSCRIPTS
def merge_transcripts():
    print("Merging all transcripts into one file...")

    txt_output = os.path.join(TRANSCRIPT_DIR, "full_transcript.txt")

    with open(txt_output, "w", encoding="utf-8") as outfile:
        files = sorted([f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".json")])

        for f in files:
            path = os.path.join(TRANSCRIPT_DIR, f)
            with open(path, "r") as jf:
                data = json.load(jf)
                if data.get("text"):
                    outfile.write(data["text"] + "\n\n")

    print("   Transcript merged -> full_transcript.txt\n")


# MAIN PIPELINE
def run():
    print("Starting Audio Processing Pipeline...\n")

    files = [f for f in os.listdir(RAW_DIR) if f.endswith((".mp3", ".wav"))]

    if not files:
        print("No audio files found inside dataset/raw_data/")
        return

    for f in files:
        print(f"\n=======================================")
        print(f"Processing file: {f}")
        print(f"=======================================\n")

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

    merge_transcripts()

    print("All processing completed successfully.")


run()
