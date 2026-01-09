from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
from rppg import extract_rppg

app = FastAPI(
    title="rPPG Deepfake Detector",
    description="Research-grade deepfake detection using physiological signals",
    version="2.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "rPPG Deepfake Detector",
        "version": "2.0",
        "description": "Upload video to /detect endpoint",
        "features": [
            "Skin-pixel-only extraction (YCrCb segmentation)",
            "POS method for rPPG",
            "Sliding-window temporal analysis",
            "Cross-ROI coherence checking",
            "Physiological rule-based validation",
            "Suspicious segment detection"
        ]
    }

@app.post("/detect")
async def detect_video(file: UploadFile = File(...)):
    """
    Single endpoint for complete deepfake detection
    
    Returns:
    - verdict: REAL or FAKE
    - confidence: 0-1 score
    - physiological_metrics: HR, SNR, etc.
    - temporal_analysis: sliding window results
    - suspicious_segments: array of {start, end} times
    - visualizations: base64 encoded plots
    """
    
    # Validate file type
    allowed_ext = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file. Allowed: {', '.join(allowed_ext)}"
        )
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Process video
    try:
        result, error = extract_rppg(filepath)
    except Exception as e:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    # Handle errors
    if error:
        if os.path.exists(filepath):
            os.remove(filepath)
        return {
            "status": "error",
            "verdict": "INCONCLUSIVE",
            "error": error.get("error", "Unknown error"),
            "details": error
        }
    
    # Clean up video file (optional - comment out to keep videos)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Return complete result
    result["status"] = "success"
    result["filename"] = filename
    result["timestamp"] = datetime.now().isoformat()
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)