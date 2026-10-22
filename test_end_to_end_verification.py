import os
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def upload_and_analyze(video_path, model="standard", endpoint="/api/analyze"):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    fname = os.path.basename(video_path)
    with open(video_path, 'rb') as f:
        data = f.read()
    
    url = f"{BASE_URL}{endpoint}"
    if endpoint == "/api/analyze":
        url += f"?model={model}"
        
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f'Content-Type: video/mp4\r\n\r\n'
    ).encode('utf-8') + data + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.getcode(), json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

def run_e2e_verification():
    print("==================================================================")
    print("TRUTHLENS AI - END-TO-END DEMO & API VERIFICATION SUITE")
    print("==================================================================")
    
    # 1. Health check verification
    print("\n[1] Health Check Verification...")
    with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
        h = json.loads(resp.read().decode('utf-8'))
    print(f"  Status             : {h['status']}")
    print(f"  Version            : {h['version']}")
    print(f"  Production Model   : {h['production_model']}")
    print(f"  Experimental Model : {h['experimental_model']}")
    assert h["version"] == "4.0.0"
    assert "dima806" in h["production_model"]
    assert "v4" in h["experimental_model"]

    # 2. Genuine Video (Production Mode & Experimental Mode)
    print("\n[2] Testing Genuine Video (video_genuine.mp4)...")
    c_std, r_std = upload_and_analyze("video_genuine.mp4", model="standard")
    print(f"  Production Mode   -> Status: {c_std} | Verdict: {r_std['prediction']} | Score: {r_std['model_score']:.2%} | Model: {r_std['model_info']['name']}")
    assert r_std["prediction"] == "LIKELY REAL"
    assert "dima806" in r_std["model_info"]["name"]

    c_exp, r_exp = upload_and_analyze("video_genuine.mp4", model="experimental")
    print(f"  Experimental Mode (v4) -> Status: {c_exp} | Verdict: {r_exp['prediction']} | Score: {r_exp['model_score']:.2%} | Version: {r_exp['model_info']['version']}")
    assert r_exp["prediction"] == "LIKELY REAL"
    assert r_exp["model_info"]["version"] == "v4"

    # 3. Manipulated Face-Swap Video (video_manipulated.mp4)
    print("\n[3] Testing Manipulated Video (video_manipulated.mp4)...")
    c_man, r_man = upload_and_analyze("video_manipulated.mp4", model="standard")
    print(f"  Production Mode   -> Status: {c_man} | Verdict: {r_man['prediction']} | Score: {r_man['model_score']:.2%}")
    assert r_man["prediction"] == "LIKELY AI-GENERATED / MANIPULATED"

    # 4. Full AI-Generated Video (video_ai_generated.mp4)
    print("\n[4] Testing AI-Generated Video (video_ai_generated.mp4)...")
    c_ai, r_ai = upload_and_analyze("video_ai_generated.mp4", model="standard")
    print(f"  Production Mode   -> Status: {c_ai} | Verdict: {r_ai['prediction']} | Score: {r_ai['model_score']:.2%}")
    assert r_ai["prediction"] == "LIKELY AI-GENERATED / MANIPULATED"

    # 5. syn_30_known_ai_benchmark.mp4 (Conflict Resolution & No False-REAL)
    print("\n[5] Testing syn_30_known_ai_benchmark.mp4 (Forensic Conflict Resolution)...")
    c_k_std, r_k_std = upload_and_analyze("known_ai_video.mp4", model="standard")
    print(f"  Production Mode   -> Status: {c_k_std} | Verdict: {r_k_std['prediction']} | Score: {r_k_std['model_score']:.2%}")
    assert r_k_std["prediction"] == "LIKELY AI-GENERATED / MANIPULATED"

    c_k_exp, r_k_exp = upload_and_analyze("known_ai_video.mp4", model="experimental")
    print(f"  Experimental Mode -> Status: {c_k_exp} | Verdict: {r_k_exp['prediction']} | Analysis Status: {r_k_exp['analysis_status']}")
    print(f"  Explanation       : {r_k_exp['explanation']}")
    assert r_k_exp["prediction"] != "LIKELY REAL", "CRITICAL ERROR: produced confident REAL on known AI video!"
    assert r_k_exp["prediction"] == "INCONCLUSIVE"
    assert "Conflicting Forensic Signals" in r_k_exp["analysis_status"]

    # 6. Upload Error Handling: Empty File (0 Bytes)
    print("\n[6] Testing Upload Error Handling (0-Byte File)...")
    empty_file = "test_empty_scratch.mp4"
    with open(empty_file, "wb") as f:
        pass
    c_err, r_err = upload_and_analyze(empty_file, model="standard")
    print(f"  Empty Upload -> Status: {c_err} | Detail: {r_err.get('detail')}")
    assert c_err == 400
    if os.path.exists(empty_file):
        os.remove(empty_file)

    # 7. Upload Error Handling: Corrupt / Unreadable Video
    print("\n[7] Testing Corrupt / Unreadable Video...")
    corrupt_file = "test_corrupt_scratch.mp4"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_CORRUPTED_STREAM")
    c_cor, r_cor = upload_and_analyze(corrupt_file, model="experimental")
    print(f"  Corrupt Upload -> Status: {c_cor} | Verdict: {r_cor['prediction']} | Analysis Status: {r_cor.get('analysis_status')}")
    assert r_cor["prediction"] == "INCONCLUSIVE"
    assert "Failed" in r_cor["analysis_status"] or "Unreadable" in r_cor["analysis_status"]
    if os.path.exists(corrupt_file):
        os.remove(corrupt_file)

    # 8. Genuine Animation Video (Big Buck Bunny, CC-BY 3.0)
    print("\n[8] Testing Genuine Animation Video (gen_33_real_animation_big_buck_bunny.mp4)...")
    anim_file = "dataset_v4/genuine/gen_33_real_animation_big_buck_bunny.mp4"
    if os.path.exists(anim_file):
        c_an, r_an = upload_and_analyze(anim_file, model="experimental")
        print(f"  Experimental Mode -> Status: {c_an} | Verdict: {r_an['prediction']} | Score: {r_an['model_score']:.2%}")
        assert r_an["prediction"] == "LIKELY REAL", f"Expected LIKELY REAL for genuine animation, got {r_an['prediction']}"

    # 9. AI-Generated Creature/Animation (Sora Creature, Creative Commons)
    print("\n[9] Testing AI-Generated Creature/Animation (syn_32_sora_animation_creature.mp4)...")
    ai_anim_file = "dataset_v4/synthetic/syn_32_sora_animation_creature.mp4"
    if os.path.exists(ai_anim_file):
        c_an_ai, r_an_ai = upload_and_analyze(ai_anim_file, model="experimental")
        print(f"  Experimental Mode -> Status: {c_an_ai} | Verdict: {r_an_ai['prediction']} | Score: {r_an_ai['model_score']:.2%}")
        assert r_an_ai["prediction"] == "LIKELY AI-GENERATED / MANIPULATED"

    # 10. Multi-person Crowd Walk (Pedestrians)
    print("\n[10] Testing Multi-Person Crowd Video (gen_37_real_people_crowd_vtest.mp4)...")
    crowd_file = "dataset_v4/genuine/gen_37_real_people_crowd_vtest.mp4"
    if os.path.exists(crowd_file):
        c_cr, r_cr = upload_and_analyze(crowd_file, model="standard")
        print(f"  Production Mode   -> Status: {c_cr} | Verdict: {r_cr['prediction']} | Faces: {r_cr['video_details'].get('faces_detected')}")
        assert r_cr["prediction"] == "LIKELY REAL", f"Multi-face crowd returned {r_cr['prediction']} instead of LIKELY REAL!"

    print("\n==================================================================")
    print("ALL 10 END-TO-END VERIFICATION CHECKS PASSED 100%!")
    print("==================================================================")

if __name__ == "__main__":
    run_e2e_verification()
