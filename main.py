from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
from rppg import extract_rppg
import json

app = FastAPI(
    title="Advanced rPPG Deepfake Detector",
    description="Physiological signal-based deepfake video detection using rPPG analysis",
)

# CORS middleware for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):

    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    # Process video
    try:
        result, error = extract_rppg(video_path)
    except Exception as e:
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    # Handle errors
    if error:
        response = {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.0,
            "reason": error.get("error", "Unknown error"),
            "error_details": error,
            "status": "error"
        }
        
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return JSONResponse(content=response, status_code=200)
    
    # Prepare successful response
    response = {
        "status": "success",
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "confidence_percentage": f"{result['confidence'] * 100:.1f}%",
        "reason": result["reason"],
        "analysis": {
            "method_used": result["method_used"],
            "fps": result["fps"],
            "total_frames": result["total_frames"],
            "used_frames": result["used_frames"],
            "frame_usage_rate": f"{(result['used_frames'] / result['total_frames'] * 100):.1f}%",
            "motion_penalty": result["motion_penalty"]
        },
        "physiological_metrics": {
            "heart_rate_bpm": result["features"]["heart_rate_bpm"],
            "signal_strength": result["features"]["std"],
            "snr": result["features"]["snr"],
            "spectral_purity": result["features"]["spectral_purity"],
            "peak_regularity": result["features"]["peak_regularity"]
        },
        "advanced_metrics": {
            "hr_power_ratio": result["features"]["hr_power_ratio"],
            "spectral_entropy": result["features"]["spectral_entropy"],
            "num_peaks": result["features"]["num_peaks"],
            "rms": result["features"]["rms"]
        },
        "visualizations": {
            "waveform_plot": f"/plot/waveform",
            "spectrum_plot": f"/plot/spectrum"
        },
        "suspicious_segments": result.get("suspicious_segments"),
        "timestamp": datetime.now().isoformat(),
        "filename": safe_filename
    }
    
    # Save results to file
    result_path = os.path.join(RESULTS_DIR, f"{timestamp}_result.json")
    with open(result_path, 'w') as f:
        json.dump(response, f, indent=2)
    

    return response

@app.delete("/cleanup")
async def cleanup_files():

    
    deleted_count = 0
    
    # Clean uploads
    if os.path.exists(UPLOAD_DIR):
        for file in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, file))
                deleted_count += 1
            except:
                pass
    
    # Clean plots
    if os.path.exists("plots"):
        for file in os.listdir("plots"):
            try:
                os.remove(os.path.join("plots", file))
                deleted_count += 1
            except:
                pass
    
    return {
        "status": "success",
        "deleted_files": deleted_count,
        "message": "Cleanup completed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)