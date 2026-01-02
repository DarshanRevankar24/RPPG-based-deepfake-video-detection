from fastapi import FastAPI, UploadFile, File
import shutil
import os
from rppg import extract_rppg

app = FastAPI(title="Physiological rPPG Deepfake Detector")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result, error = extract_rppg(video_path)

    if error:
        return {
            "verdict": "FAKE",
            "reason": error["error"],
            "fps": error["fps"],
            "total_frames": error["total_frames"],
            "used_frames": error["used_frames"],
            "suspicious_rppg_segments": None
        }

    THRESHOLD = 0.02
    verdict = "REAL" if result["strength"] >= THRESHOLD else "FAKE"

    return {
        "verdict": verdict,
        "reason": "Valid physiological heartbeat detected" if verdict == "REAL" else "No stable rPPG detected",
        "heartbeat_strength": result["strength"],
        "confidence_score": result["confidence"],
        "fps": result["fps"],
        "total_frames": result["total_frames"],
        "used_frames": result["used_frames"],
        "rppg_waveform_plot": result["waveform_plot"],
        "fft_spectrum_plot": result["fft_plot"],
        "suspicious_rppg_segments": result["suspicious_segments"]
    }
