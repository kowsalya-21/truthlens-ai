import os
import sys
import json
import shutil
import urllib.request
import ssl
import cv2
import numpy as np

DATASET_DIR = "E:/truthlens-ai/dataset_v3"
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")
GENUINE_DIR = os.path.join(DATASET_DIR, "genuine")
SYNTHETIC_DIR = os.path.join(DATASET_DIR, "synthetic")
TEMP_DIR = os.path.join(DATASET_DIR, "_temp_downloads")

os.makedirs(GENUINE_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def check_disk_space():
    total, used, free = shutil.disk_usage("E:/")
    free_gb = free / (1024**3)
    print(f"[Storage Pre-Check] Drive E: Free Space: {free_gb:.2f} GB")
    if free_gb < 5.0:
        raise RuntimeError(f"Insufficient disk space on Drive E: ({free_gb:.2f} GB available, need >= 5 GB)")
    return free_gb

def normalize_and_save_video(src_path, dst_path, max_frames=48, max_dim=480):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False, "Failed to open source video with OpenCV"
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or fps > 120:
        fps = 24.0
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        return False, f"Invalid dimensions: {w}x{h}"
        
    scale = min(1.0, max_dim / max(w, h))
    new_w = int(round(w * scale / 2.0) * 2) # ensure even
    new_h = int(round(h * scale / 2.0) * 2) # ensure even
    
    # Read frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if len(frames) >= 120: # read at most 120 frames to conserve memory
            break
    cap.release()
    
    if len(frames) < 16:
        # Loop frames if clip is short
        if len(frames) == 0:
            return False, "Zero valid frames read"
        while len(frames) < 16:
            frames.extend(frames[:min(len(frames), 16 - len(frames))])
            
    # Sample up to max_frames
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frames = [frames[min(i, len(frames)-1)] for i in indices]
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, min(fps, 30.0), (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        out.write(f)
    out.release()
    
    # Verify written video
    verify_cap = cv2.VideoCapture(dst_path)
    v_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    verify_cap.release()
    
    if v_frames < 16:
        return False, f"Output video has only {v_frames} frames (expected >= 16)"
        
    return True, f"Saved {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]})"

def acquire_dataset():
    check_disk_space()
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    print(f"Beginning dataset acquisition for {len(manifest)} items...")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    success_count = 0
    skipped_count = 0
    failed_items = []
    
    for idx, item in enumerate(manifest, 1):
        iid = item["id"]
        rel_path = item["filename"]
        target_path = os.path.join(DATASET_DIR, rel_path)
        url = item["source_url"]
        
        # Check if already processed and valid
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            chk_cap = cv2.VideoCapture(target_path)
            cnt = int(chk_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            chk_cap.release()
            if cnt >= 16:
                print(f"[{idx:02d}/60] [CACHED] {iid:<10} | {rel_path} ({cnt} frames, {os.path.getsize(target_path)/1024:.1f} KB)")
                skipped_count += 1
                success_count += 1
                continue
                
        print(f"[{idx:02d}/60] [ACQUIRING] {iid:<10} | {item['label']} | {item['category']}...")
        
        # Determine source
        if url.upper().startswith("LOCAL:"):
            local_src = url.split(":", 1)[1].strip()
            if not os.path.exists(local_src):
                failed_items.append((iid, rel_path, f"Local file missing: {local_src}"))
                print(f"  FAILED: Local file missing: {local_src}")
                continue
            ok, msg = normalize_and_save_video(local_src, target_path)
            if ok:
                success_count += 1
                print(f"  SUCCESS: {msg} -> {os.path.getsize(target_path)/1024:.1f} KB")
            else:
                failed_items.append((iid, rel_path, msg))
                print(f"  FAILED: {msg}")
        else:
            # Remote download
            ext = os.path.splitext(url.split("?")[0])[1]
            if not ext or len(ext) > 5:
                ext = ".mp4"
            temp_path = os.path.join(TEMP_DIR, f"temp_{iid}{ext}")
            
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TruthLensResearch/1.0"}
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp, open(temp_path, "wb") as out_f:
                    shutil.copyfileobj(resp, out_f)
            except Exception as e:
                failed_items.append((iid, rel_path, f"Download error: {e}"))
                print(f"  DOWNLOAD ERROR: {e}")
                continue
                
            ok, msg = normalize_and_save_video(temp_path, target_path)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    
            if ok:
                success_count += 1
                print(f"  SUCCESS: {msg} -> {os.path.getsize(target_path)/1024:.1f} KB")
            else:
                failed_items.append((iid, rel_path, msg))
                print(f"  FAILED: {msg}")
                
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        
    print("\n================ ACQUISITION SUMMARY ================")
    print(f"Total manifest items : {len(manifest)}")
    print(f"Successfully acquired: {success_count} / {len(manifest)}")
    print(f"Already cached       : {skipped_count}")
    print(f"Failed acquisitions  : {len(failed_items)}")
    
    if failed_items:
        print("Failure details:")
        for f in failed_items:
            print(f"  ID: {f[0]} | Path: {f[1]} | Error: {f[2]}")
        sys.exit(1)
    else:
        print("ALL 60 VIDEOS ACQUIRED AND STANDARDIZED WITH >= 16 FRAMES!")

if __name__ == "__main__":
    acquire_dataset()
