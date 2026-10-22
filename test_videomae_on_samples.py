import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEModel

MODEL_NAME = "MCG-NJU/videomae-base"
CHECKPOINT_PATH = "e:/truthlens-ai/backend/weights/videomae_general_head.pt"

class SpatiotemporalAuthenticityHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=128, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def sample_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: total_frames = 60
    indices = [int(i * (total_frames - 1) / max(1, num_frames - 1)) for i in range(num_frames)]
    indices = sorted(list(set(indices)))
    while len(indices) < num_frames:
        indices.append(indices[-1])
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    cap.release()
    return frames

print("Loading VideoMAE Backbone and Trained Authenticity Head...")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
backbone = VideoMAEModel.from_pretrained(MODEL_NAME)
backbone.eval()

# Reload saved weights
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
head = SpatiotemporalAuthenticityHead(**{k: v for k, v in checkpoint["config"].items() if k in ["in_dim", "hidden_dim", "out_dim", "dropout"]})
head.load_state_dict(checkpoint["state_dict"])
head.eval()

videos = ["video_genuine.mp4", "video_manipulated.mp4", "video_ai_generated.mp4"]

print("\n" + "="*70)
print("EXPERIMENTAL VIDEOMAE INFERENCE ON PRODUCTION SAMPLE VIDEOS")
print("="*70)

for v in videos:
    frames = sample_video_frames(v, num_frames=16)
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = backbone(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        logits = head(emb)
        probs = torch.softmax(logits, dim=-1)
        
    p_real = float(probs[0, 0].item())
    p_fake = float(probs[0, 1].item())
    
    if p_fake >= 0.55:
        pred = "LIKELY AI-GENERATED / MANIPULATED"
    elif p_fake <= 0.45:
        pred = "LIKELY REAL"
    else:
        pred = "INCONCLUSIVE"
        
    print(f"Video           : {v}")
    print(f"Prediction      : {pred}")
    print(f"Fake Probability: {p_fake:.4f} ({p_fake:.1%})")
    print(f"Real Probability: {p_real:.4f} ({p_real:.1%})")
    print(f"Logits          : Real={logits[0, 0].item():.4f}, Fake={logits[0, 1].item():.4f}")
    print("-" * 70)
