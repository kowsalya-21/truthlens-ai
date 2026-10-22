import os
import sys
import json
import urllib.request
import cv2

EVAL_DIR = "e:/truthlens-ai/eval_dataset"
GEN_DIR = os.path.join(EVAL_DIR, "genuine")
SYN_DIR = os.path.join(EVAL_DIR, "synthetic")
os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(SYN_DIR, exist_ok=True)

def trim_to_clip(src_path, dst_path, target_frames=75, target_h=480):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"Error opening {src_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scale = target_h / float(h) if h > target_h else 1.0
    tw = int(w * scale)
    th = int(h * scale)
    if tw % 2 != 0: tw -= 1
    if th % 2 != 0: th -= 1
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, fps, (tw, th))
    
    count = 0
    while count < target_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if (tw, th) != (w, h):
            frame = cv2.resize(frame, (tw, th))
        out.write(frame)
        count += 1
        
    cap.release()
    out.release()
    size_bytes = os.path.getsize(dst_path) if os.path.exists(dst_path) else 0
    return {
        "frames": count,
        "width": tw,
        "height": th,
        "fps": round(fps, 1),
        "duration_sec": round(count / fps, 2),
        "size_kb": round(size_bytes / 1024, 1)
    }

def download_file(url, local_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return True
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r, open(local_path, 'wb') as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return False

# Assemble 12 Genuine and 12 Synthetic clips
print("Assembling 24-clip smoke test dataset...")

# Manifest list
manifest = []

# --- 1. GENUINE CLIPS (12) ---
# Clip 1: Repository real speaking portrait
p1 = trim_to_clip("video_genuine.mp4", os.path.join(GEN_DIR, "gen_01_real_portrait_talking.mp4"))
manifest.append({
    "id": "gen_01",
    "filename": "genuine/gen_01_real_portrait_talking.mp4",
    "label": "real",
    "category": "people_portrait",
    "source": "Existing TruthLens benchmark (CC0)",
    "license": "CC0 / Public Domain",
    **p1
})

# External Intel IoT genuine camera videos
intel_base = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/"
intel_clips = [
    ("car-detection.mp4", "gen_02_real_car_traffic.mp4", "vehicles_outdoor", "Intel IoT (car-detection)"),
    ("bottle-detection.mp4", "gen_03_real_bottle_objects.mp4", "objects_indoor", "Intel IoT (bottle-detection)"),
    ("people-detection.mp4", "gen_04_real_people_street.mp4", "people_outdoor", "Intel IoT (people-detection)"),
    ("store-aisle-detection.mp4", "gen_05_real_store_interior.mp4", "indoor_store", "Intel IoT (store-aisle)"),
    ("one-by-one-person-detection.mp4", "gen_06_real_pedestrian_walk.mp4", "people_walking", "Intel IoT (pedestrians)"),
    ("person-bicycle-car-detection.mp4", "gen_07_real_bicycle_road.mp4", "traffic_activity", "Intel IoT (person-bicycle-car)"),
    ("classroom.mp4", "gen_08_real_classroom_indoor.mp4", "indoor_activity", "Intel IoT (classroom)"),
    ("head-pose-face-detection-female.mp4", "gen_09_real_female_portrait.mp4", "people_female_portrait", "Intel IoT (head-pose-female)"),
    ("head-pose-face-detection-male.mp4", "gen_10_real_male_portrait.mp4", "people_male_portrait", "Intel IoT (head-pose-male)"),
    ("bolt-detection.mp4", "gen_11_real_bolt_hardware.mp4", "objects_hardware", "Intel IoT (bolt-detection)"),
    ("fruit-and-vegetable-detection.mp4", "gen_12_real_food_vegetables.mp4", "objects_food", "Intel IoT (fruit-vegetables)")
]

for src_name, dst_name, cat, src_label in intel_clips:
    tmp_path = os.path.join(EVAL_DIR, f"temp_{src_name}")
    dst_path = os.path.join(GEN_DIR, dst_name)
    print(f"Fetching {src_name}...")
    if download_file(intel_base + src_name, tmp_path):
        meta = trim_to_clip(tmp_path, dst_path)
        if os.path.exists(tmp_path): os.remove(tmp_path)
        manifest.append({
            "id": dst_name.split('_')[1],
            "filename": f"genuine/{dst_name}",
            "label": "real",
            "category": cat,
            "source": src_label,
            "license": "Apache 2.0 / Public Sample Video",
            **meta
        })

# --- 2. SYNTHETIC & MANIPULATED CLIPS (12) ---
# Clip 1: Existing manipulated face-swap
s1 = trim_to_clip("video_manipulated.mp4", os.path.join(SYN_DIR, "syn_01_manipulated_faceswap.mp4"))
manifest.append({
    "id": "syn_01",
    "filename": "synthetic/syn_01_manipulated_faceswap.mp4",
    "label": "fake",
    "category": "manipulated_faceswap",
    "source": "Existing TruthLens benchmark (FaceForensics++)",
    "license": "Research Fair Use",
    **s1
})

# Clip 2: Existing Sora/Kling AI generation
s2 = trim_to_clip("video_ai_generated.mp4", os.path.join(SYN_DIR, "syn_02_ai_person_sora.mp4"))
manifest.append({
    "id": "syn_02",
    "filename": "synthetic/syn_02_ai_person_sora.mp4",
    "label": "fake",
    "category": "ai_generated_person",
    "source": "Existing TruthLens benchmark (Sora/Diffusion)",
    "license": "Research Fair Use",
    **s2
})

# Sora generated clips from HF ShiyumeMeguri/SoraVideo
sora_base = "https://huggingface.co/datasets/ShiyumeMeguri/SoraVideo/resolve/main/videos/"
sora_clips = [
    ("gen_01jeyf8m7xfcp9acj1x5f2vkp2.mp4", "syn_03_sora_scene_city.mp4", "ai_landscape_city"),
    ("gen_01jeyf8m8gfk58wb6va1v2tryz.mp4", "syn_04_sora_scene_nature.mp4", "ai_nature_landscape"),
    ("gen_01jeyg712efseae4b8p0qkq3rg.mp4", "syn_05_sora_scene_fantasy.mp4", "ai_fantasy_creature"),
    ("gen_01jeyg712mfb4ag4sp5cwhmbzn.mp4", "syn_06_sora_scene_water.mp4", "ai_water_physics"),
    ("gen_01jeygjv5ye57tc2f2w398awna.mp4", "syn_07_sora_scene_cinematic.mp4", "ai_cinematic_scene"),
    ("gen_01jezbngz8en196j5atjebr349.mp4", "syn_08_sora_scene_animal.mp4", "ai_animal_wildlife")
]

for src_name, dst_name, cat in sora_clips:
    tmp_path = os.path.join(EVAL_DIR, f"temp_{src_name}")
    dst_path = os.path.join(SYN_DIR, dst_name)
    print(f"Fetching Sora clip {src_name[:20]}...")
    if download_file(sora_base + src_name, tmp_path):
        meta = trim_to_clip(tmp_path, dst_path)
        if os.path.exists(tmp_path): os.remove(tmp_path)
        manifest.append({
            "id": dst_name.split('_')[1],
            "filename": f"synthetic/{dst_name}",
            "label": "fake",
            "category": cat,
            "source": "OpenAI Sora Text-to-Video (via ShiyumeMeguri/SoraVideo)",
            "license": "Research Fair Use / Evaluation",
            **meta
        })

# ByteDance synthetic physical simulation videos
byte_base = "https://huggingface.co/datasets/kevinzzz8866/ByteDance_Synthetic_Videos/resolve/main/obj_video_release1/part_1/"
byte_clips = [
    ("v6-0000020345-s1.0.mp4", "syn_09_bytedance_sim_objects.mp4", "ai_synthetic_physics_objects"),
    ("v6-0000020059-s1.0.mp4", "syn_10_bytedance_sim_motion.mp4", "ai_synthetic_physics_motion"),
    ("v6-0000020075-s1.0.mp4", "syn_11_bytedance_sim_dynamics.mp4", "ai_synthetic_physics_dynamics"),
    ("v6-0000020128-s1.0.mp4", "syn_12_bytedance_sim_fluids.mp4", "ai_synthetic_physics_fluids")
]

for src_name, dst_name, cat in byte_clips:
    tmp_path = os.path.join(EVAL_DIR, f"temp_{src_name}")
    dst_path = os.path.join(SYN_DIR, dst_name)
    print(f"Fetching ByteDance synthetic clip {src_name[:20]}...")
    if download_file(byte_base + src_name, tmp_path):
        meta = trim_to_clip(tmp_path, dst_path)
        if os.path.exists(tmp_path): os.remove(tmp_path)
        manifest.append({
            "id": dst_name.split('_')[1],
            "filename": f"synthetic/{dst_name}",
            "label": "fake",
            "category": cat,
            "source": "ByteDance Synthetic Video Benchmark (CGI Simulation)",
            "license": "Research Fair Use / Evaluation",
            **meta
        })

manifest_path = os.path.join(EVAL_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nAssembly complete! Total clips: {len(manifest)}")
real_count = sum(1 for m in manifest if m['label'] == 'real')
fake_count = sum(1 for m in manifest if m['label'] == 'fake')
total_kb = sum(m['size_kb'] for m in manifest)
print(f"Genuine: {real_count} | Synthetic: {fake_count} | Total Size: {total_kb/1024:.2f} MB")
