import os
import cv2
import torch
import numpy as np
from PIL import Image
from backend.detector import analyze_video_file, get_deepfake_model
from backend.experimental_detector import analyze_video_experimental, get_experimental_videomae, sample_uniform_frames

video_path = "e:/truthlens-ai/known_ai_video.mp4"

print("======================================================================")
print("1. VIDEO FILE PROPERTIES")
print("======================================================================")
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0
print(f"File Path    : {video_path}")
print(f"File Size    : {os.path.getsize(video_path):,} bytes")
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps}")
print(f"Total Frames : {total_frames}")
print(f"Duration     : {duration:.2f} seconds")
cap.release()

print("\n======================================================================")
print("2. STANDARD DETECTOR (Facial Specialist ViT)")
print("======================================================================")
std_result = analyze_video_file(video_path, max_samples=12)
print("Prediction        :", std_result.get("prediction"))
print("Model Score (Fake):", std_result.get("model_score"))
print("Analysis Status   :", std_result.get("analysis_status"))
print("Explanation       :", std_result.get("explanation"))
print("Video Details     :", std_result.get("video_details"))
print("Evidence Count    :", len(std_result.get("evidence", [])))
for ev in std_result.get("evidence", []):
    print(" - Evidence:", ev)
print("Frame Analysis Count:", len(std_result.get("frame_analysis", [])))
faces_found = sum(1 for f in std_result.get("frame_analysis", []) if f.get("face_detected"))
print("Frames with faces detected:", faces_found)

print("\n======================================================================")
print("3. EXPERIMENTAL DETECTOR (VideoMAE Spatiotemporal Head)")
print("======================================================================")
exp_result = analyze_video_experimental(video_path, num_frames=16)
print("Prediction        :", exp_result.get("prediction"))
print("Model Score (Fake):", exp_result.get("model_score"))
print("Analysis Status   :", exp_result.get("analysis_status"))
print("Explanation       :", exp_result.get("explanation"))
print("Metrics           :", exp_result.get("metrics"))

print("\n======================================================================")
print("4. DEEP DIVE INTO EXPERIMENTAL MODEL LOGITS & TENSORS")
print("======================================================================")
processor, backbone, head = get_experimental_videomae()
frames_pil, frame_records, total_frames, fps, width, height, duration = sample_uniform_frames(video_path, 16)
inputs = processor(frames_pil, return_tensors="pt")

with torch.no_grad():
    outputs = backbone(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    logits = head(emb)
    probs = torch.softmax(logits, dim=-1)

print("Backbone Embedding Shape:", emb.shape)
print("Backbone Embedding Norm :", torch.norm(emb).item())
print("Raw Head Logits         :", logits.numpy().tolist())
print("Softmax Probabilities   :", probs.numpy().tolist())
print(f"  Class 0 (Real) Probability: {probs[0, 0].item():.4%}")
print(f"  Class 1 (Fake) Probability: {probs[0, 1].item():.4%}")

checkpoint = torch.load("backend/weights/videomae_general_head.pt", map_location="cpu", weights_only=False)
print("Checkpoint Labels Config:", checkpoint.get("labels"))
print("Checkpoint Config       :", checkpoint.get("config"))
