import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os


mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


def bandpass_filter(signal, fps):
    low = 0.8 / (fps / 2)
    high = 3.0 / (fps / 2)
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


def get_roi_from_landmarks(frame, landmarks, indices):
    h, w, _ = frame.shape
    xs, ys = [], []
    for idx in indices:
        lm = landmarks[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))
    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)
    return frame[y1:y2, x1:x2]


def frame_motion(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    return np.mean(np.abs(flow))


def detect_abnormal_segments(signal, fps):
    window_sec = 0.4
    step_sec = 0.2
    window = int(window_sec * fps)
    step = int(step_sec * fps)

    if len(signal) < window:
        return []

    global_amp = np.max(signal) - np.min(signal)
    threshold = 1.5 * global_amp  # lower threshold to detect smaller spikes

    suspicious = []
    for start in range(0, len(signal) - window, step):
        segment = signal[start:start + window]
        local_amp = np.max(segment) - np.min(segment)
        if local_amp > threshold:
            suspicious.append({
                "start_time_sec": round(start / fps, 2),
                "end_time_sec": round((start + window) / fps, 2),
                "reason": "Abnormal rPPG amplitude spike"
            })
    return suspicious


def extract_rppg(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    forehead_signal = []
    cheek_signal = []
    total_frames = 0
    used_frames = 0
    prev_gray = None
    motion_scores = []

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
        roi_f = get_roi_from_landmarks(frame, landmarks, FOREHEAD)
        roi_l = get_roi_from_landmarks(frame, landmarks, LEFT_CHEEK)
        roi_r = get_roi_from_landmarks(frame, landmarks, RIGHT_CHEEK)

        if roi_f.size == 0 or roi_l.size == 0 or roi_r.size == 0:
            continue

        forehead_signal.append(np.mean(roi_f[:, :, 1]))
        cheek_signal.append((np.mean(roi_l[:, :, 1]) + np.mean(roi_r[:, :, 1])) / 2)
        used_frames += 1

    cap.release()

    if used_frames < fps * 3:
        return None, {
            "error": "Insufficient stable facial frames",
            "fps": fps,
            "total_frames": total_frames,
            "used_frames": used_frames
        }

    motion_penalty = np.mean(motion_scores) if motion_scores else 0
    fused = 0.6 * np.array(forehead_signal) + 0.4 * np.array(cheek_signal)
    filtered = bandpass_filter(fused, fps)

    strength = np.std(filtered)
    fft_vals = np.abs(fft(filtered))
    freqs = fftfreq(len(fft_vals), d=1/fps)
    hr_band = (freqs > 0.8) & (freqs < 3.0)
    sqi = np.sum(fft_vals[hr_band]) / np.sum(fft_vals)
    confidence = min((strength * sqi) / (1 + motion_penalty), 1.0)

    suspicious_segments = detect_abnormal_segments(filtered, fps)

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
        "fft_plot": fft_path,
        "suspicious_segments": suspicious_segments if suspicious_segments else None
    }, None
