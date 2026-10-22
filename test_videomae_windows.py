import cv2
import torch
from PIL import Image
from backend.experimental_detector import get_experimental_videomae

video_path = "e:/truthlens-ai/known_ai_video.mp4"
processor, backbone, head = get_experimental_videomae()

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

print(f"Total Frames: {total_frames}, FPS: {fps}")

# Test multiple 16-frame consecutive windows throughout the video
windows = [
    ("Start (frames 0..15)", range(0, 16)),
    ("Early (frames 50..65)", range(50, 66)),
    ("Middle (frames 150..165)", range(150, 166)),
    ("Transition (frames 280..295)", range(280, 296)),
    ("Late (frames 430..445)", range(430, 446)),
    ("End (frames 500..515)", range(500, 516)),
    ("Uniform 16 (stride 39 across whole video)", [int(i * (total_frames - 1) / 15) for i in range(16)])
]

for label, indices in windows:
    frames = []
    for f_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        else:
            frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = backbone(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        logits = head(emb)
        probs = torch.softmax(logits, dim=-1)
        
    p_real = probs[0, 0].item()
    p_fake = probs[0, 1].item()
    print(f"Window: {label:<45} | Logits: [{logits[0,0].item():.2f}, {logits[0,1].item():.2f}] | Real: {p_real:.2%}, Fake: {p_fake:.2%}")

cap.release()
