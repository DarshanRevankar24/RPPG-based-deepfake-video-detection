import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

# -------------------- MediaPipe --------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# -------------------- Filters --------------------
def bandpass_filter(signal, fps):
    low = 0.8 / (fps / 2)
    high = 3.0 / (fps / 2)
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)

# -------------------- ROI extraction --------------------
def get_roi_from_landmarks(frame, landmarks, indices):
    h, w, _ = frame.shape
    xs, ys = [], []

    for idx in indices:
        lm = landmarks[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)

    roi = frame[y1:y2, x1:x2]
    return roi

# -------------------- Motion Estimation --------------------
def frame_motion(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    return np.mean(np.abs(flow))

# -------------------- Main rPPG Extraction --------------------
def extract_rppg(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    forehead_signal = []
    cheek_signal = []

    total_frames = 0
    used_frames = 0

    prev_gray = None
    motion_scores = []

    # Landmark indices
    FOREHEAD = [10, 67, 69, 104]
    LEFT_CHEEK = [234, 93, 132]
    RIGHT_CHEEK = [454, 323, 361]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        if prev_gray is not None:
            motion_scores.append(frame_motion(prev_gray, gray))
        prev_gray = gray

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        roi_forehead = get_roi_from_landmarks(frame, landmarks, FOREHEAD)
        roi_l = get_roi_from_landmarks(frame, landmarks, LEFT_CHEEK)
        roi_r = get_roi_from_landmarks(frame, landmarks, RIGHT_CHEEK)

        if roi_forehead.size == 0 or roi_l.size == 0 or roi_r.size == 0:
            continue

        # Green channel extraction
        g_forehead = np.mean(roi_forehead[:, :, 1])
        g_cheek = (np.mean(roi_l[:, :, 1]) + np.mean(roi_r[:, :, 1])) / 2

        forehead_signal.append(g_forehead)
        cheek_signal.append(g_cheek)
        used_frames += 1

    cap.release()

    if used_frames < fps * 3:
        return None, {
            "error": "Insufficient stable facial frames",
            "fps": fps,
            "total_frames": total_frames,
            "used_frames": used_frames
        }

    # -------------------- Motion rejection --------------------
    motion_scores = np.array(motion_scores)
    motion_penalty = np.mean(motion_scores)

    # -------------------- Signal fusion --------------------
    forehead_signal = np.array(forehead_signal)
    cheek_signal = np.array(cheek_signal)

    fused_signal = 0.6 * forehead_signal + 0.4 * cheek_signal
    filtered = bandpass_filter(fused_signal, fps)

    # -------------------- Signal Quality --------------------
    strength = np.std(filtered)
    fft_vals = np.abs(fft(filtered))
    freqs = fftfreq(len(fft_vals), d=1/fps)

    hr_band = (freqs > 0.8) & (freqs < 3.0)
    sqi = np.sum(fft_vals[hr_band]) / np.sum(fft_vals)

    confidence = min((strength * sqi) / (1 + motion_penalty), 1.0)

    # -------------------- Save plots --------------------
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(filtered)
    plt.title("Fused rPPG Signal")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    waveform_path = "plots/rppg_waveform.png"
    plt.savefig(waveform_path)
    plt.close()

    plt.figure()
    plt.plot(freqs, fft_vals)
    plt.xlim(0, 5)
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    fft_path = "plots/fft_spectrum.png"
    plt.savefig(fft_path)
    plt.close()

    return {
        "strength": float(strength),
        "confidence": float(confidence),
        "fps": fps,
        "total_frames": total_frames,
        "used_frames": used_frames,
        "waveform_plot": waveform_path,
        "fft_plot": fft_path
    }, None
