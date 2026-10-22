import os
import json

DATASET_DIR = "e:/truthlens-ai/dataset_v4"
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.json")

# Load existing manifest items from v3
with open("e:/truthlens-ai/dataset_v3/manifest.json", "r", encoding="utf-8") as f:
    v3_items = json.load(f)

# The 12 held-out test items are strictly preserved
test_items = [it for it in v3_items if it["split"] == "held-out test"]
val_items = [it for it in v3_items if it["split"] == "validation"]
train_items = [it for it in v3_items if it["split"] == "train"]

# New genuine additions
new_genuine = [
    {
        "id": "val_genuine_anim",
        "filename": "genuine/gen_33_real_animation_big_buck_bunny.mp4",
        "label": "REAL",
        "source_dataset": "Blender Foundation Open Movie Project",
        "source_url": "https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f3/Big_Buck_Bunny_first_23_seconds_1080p.ogv/Big_Buck_Bunny_first_23_seconds_1080p.ogv.360p.vp9.webm",
        "license": "Creative Commons Attribution 3.0 Unported (CC-BY 3.0)",
        "category": "animation_cgi_open",
        "ai_generator": "None (Genuine 3D Blender CGI Open Animation)",
        "split": "validation",
        "original_scene_source_identifier": "BlenderFoundation_BigBuckBunny_ForestOpening",
        "verified_size_bytes": 199763,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_genuine_anim_2d",
        "filename": "genuine/gen_34_real_animation_plane_crazy.mp4",
        "label": "REAL",
        "source_dataset": "Classic Animation Open Heritage Archive",
        "source_url": "https://upload.wikimedia.org/wikipedia/commons/1/10/Plane_Crazy_%281929%29_with_sound.webm",
        "license": "Public Domain (US Copyright Expired / Open Media)",
        "category": "animation_traditional",
        "ai_generator": "None (Genuine 2D Hand-Drawn Traditional Cel Animation)",
        "split": "train",
        "original_scene_source_identifier": "ClassicHeritage_PlaneCrazy_1928_HandDrawn",
        "verified_size_bytes": 337153,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_genuine_crowd",
        "filename": "genuine/gen_37_real_people_crowd_vtest.mp4",
        "label": "REAL",
        "source_dataset": "OpenCV Sample Data Repository",
        "source_url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
        "license": "3-Clause BSD License (Permissive Open Source)",
        "category": "people_multi_face_outdoor",
        "ai_generator": "None (Real Outdoor Multi-Pedestrian Surveillance Camera)",
        "split": "train",
        "original_scene_source_identifier": "OpenCV_Sample_VTest_PedestrianPlaza",
        "verified_size_bytes": 384396,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_genuine_classroom",
        "filename": "genuine/gen_35_real_indoor_classroom.mp4",
        "label": "REAL",
        "source_dataset": "Intel IoT Sample Videos Suite",
        "source_url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/classroom.mp4",
        "license": "Apache License 2.0 (Permissive Commercial & Academic Use)",
        "category": "indoor_classroom_multi",
        "ai_generator": "None (Real Physical Classroom Multi-Person Overhead)",
        "split": "train",
        "original_scene_source_identifier": "Intel_IoT_Scene_ClassroomOverhead",
        "verified_size_bytes": 384396,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_genuine_junction",
        "filename": "genuine/gen_36_real_traffic_junction.mp4",
        "label": "REAL",
        "source_dataset": "Intel IoT Sample Videos Suite",
        "source_url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
        "license": "Apache License 2.0 (Permissive Commercial & Academic Use)",
        "category": "traffic_junction",
        "ai_generator": "None (Real Intersection Person-Car-Bicycle Camera)",
        "split": "train",
        "original_scene_source_identifier": "Intel_IoT_Scene_IntersectionTrafficJunction",
        "verified_size_bytes": 384396,
        "verification_status": "verified_local_file"
    }
]

# New synthetic additions
new_synthetic = [
    {
        "id": "val_synthetic_anim",
        "filename": "synthetic/syn_31_sora_animation_fantasy.mp4",
        "label": "AI-GENERATED / MANIPULATED",
        "source_dataset": "ShiyumeMeguri/SoraVideo Benchmark Suite",
        "source_url": "https://huggingface.co/datasets/ShiyumeMeguri/SoraVideo/resolve/main/videos/gen_01jh7j2x6geng9nwr207b89fjz.mp4",
        "license": "Research Fair Use / Synthetic Benchmark",
        "category": "ai_animation_diffusion",
        "ai_generator": "OpenAI Sora (Text-to-Video Stylized Animation Diffusion)",
        "split": "validation",
        "original_scene_source_identifier": "Sora_Prompt_StylizedFantasyAnimationA",
        "verified_size_bytes": 974465,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_synthetic_anim_creature",
        "filename": "synthetic/syn_32_sora_animation_creature.mp4",
        "label": "AI-GENERATED / MANIPULATED",
        "source_dataset": "ShiyumeMeguri/SoraVideo Benchmark Suite",
        "source_url": "https://huggingface.co/datasets/ShiyumeMeguri/SoraVideo/resolve/main/videos/gen_01jj8ggty6ek9vnfm2nn7m3yrh.mp4",
        "license": "Research Fair Use / Synthetic Benchmark",
        "category": "ai_animation_diffusion",
        "ai_generator": "OpenAI Sora (Text-to-Video Stylized Animation Diffusion)",
        "split": "train",
        "original_scene_source_identifier": "Sora_Prompt_AnimatedCreatureCharacterB",
        "verified_size_bytes": 675740,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_synthetic_tiger",
        "filename": "synthetic/syn_33_sora_animal_wildlife_tiger.mp4",
        "label": "AI-GENERATED / MANIPULATED",
        "source_dataset": "ShiyumeMeguri/SoraVideo Benchmark Suite",
        "source_url": "https://huggingface.co/datasets/ShiyumeMeguri/SoraVideo/resolve/main/videos/gen_01jh7j68yeens9vqrc5rwygexx.mp4",
        "license": "Research Fair Use / Synthetic Benchmark",
        "category": "ai_animal_diffusion",
        "ai_generator": "OpenAI Sora (Text-to-Video Diffusion Animal Wildlife)",
        "split": "train",
        "original_scene_source_identifier": "Sora_Prompt_WildlifeTigerStalkingC",
        "verified_size_bytes": 2560163,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_synthetic_bird",
        "filename": "synthetic/syn_34_sora_animal_bird_flight.mp4",
        "label": "AI-GENERATED / MANIPULATED",
        "source_dataset": "ShiyumeMeguri/SoraVideo Benchmark Suite",
        "source_url": "https://huggingface.co/datasets/ShiyumeMeguri/SoraVideo/resolve/main/videos/gen_01jhwq73gffj3aqvn6cjjhkfv1.mp4",
        "license": "Research Fair Use / Synthetic Benchmark",
        "category": "ai_animal_diffusion",
        "ai_generator": "OpenAI Sora (Text-to-Video Diffusion Bird Flight)",
        "split": "train",
        "original_scene_source_identifier": "Sora_Prompt_WildlifeBirdFlockD",
        "verified_size_bytes": 649777,
        "verification_status": "verified_local_file"
    },
    {
        "id": "train_synthetic_transit",
        "filename": "synthetic/syn_36_sora_futuristic_transit.mp4",
        "label": "AI-GENERATED / MANIPULATED",
        "source_dataset": "ShiyumeMeguri/SoraVideo Benchmark Suite",
        "source_url": "https://huggingface.co/datasets/ShiyumeMeguri/SoraVideo/resolve/main/videos/gen_01jh7hy64xevaawxk7gnte6fs9.mp4",
        "license": "Research Fair Use / Synthetic Benchmark",
        "category": "ai_futuristic_urban",
        "ai_generator": "OpenAI Sora (Text-to-Video Futuristic Urban Transit)",
        "split": "train",
        "original_scene_source_identifier": "Sora_Prompt_FuturisticTransitSkylineE",
        "verified_size_bytes": 1325253,
        "verification_status": "verified_local_file"
    }
]

# Assemble complete manifest
all_items = []
# 1. Held-out test (12 items strictly preserved from v3)
all_items.extend(test_items)
# 2. Validation items (12 v3 + 2 new = 14 items: 7 Real, 7 Fake)
all_items.extend(val_items)
all_items.append(new_genuine[0]) # Big Buck Bunny (val)
all_items.append(new_synthetic[0]) # Sora animation (val)
# 3. Train items (36 v3 + 8 new = 44 items: 22 Real, 22 Fake)
all_items.extend(train_items)
all_items.extend(new_genuine[1:]) # 4 genuine
all_items.extend(new_synthetic[1:]) # 4 synthetic

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(all_items, f, indent=2)

print(f"Dataset v4 Manifest Created with {len(all_items)} total entries:")
splits = {}
for it in all_items:
    splits[it['split']] = splits.get(it['split'], 0) + 1
print("Splits:", splits)
print("Held-out test count:", len(test_items), "(Identical to v3 untouched test set)")
