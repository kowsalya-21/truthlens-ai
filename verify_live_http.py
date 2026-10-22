import os
import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def upload_video(video_path, model="standard", endpoint="/api/analyze"):
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            return resp.getcode(), json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

def test_live_server():
    print("=" * 70)
    print("LIVE SERVER HTTP VERIFICATION (http://127.0.0.1:8000)")
    print("=" * 70)

    # 1. Health
    with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
        h = json.loads(resp.read().decode('utf-8'))
    print(f"\n[1] Health Check: status={h['status']}, v{h['version']}")
    print(f"    Production: {h['production_model']}")
    print(f"    Experimental: {h['experimental_model']}")

    # 2. Genuine Video (Production & Experimental)
    print("\n[2] Testing video_genuine.mp4...")
    code_std, res_std = upload_video("video_genuine.mp4", model="standard")
    print(f"    Standard     -> HTTP {code_std} | Verdict: {res_std['prediction']} | Score: {res_std['model_score']:.2%}")
    assert res_std['prediction'] == "LIKELY REAL"

    code_exp, res_exp = upload_video("video_genuine.mp4", model="experimental")
    print(f"    Experimental -> HTTP {code_exp} | Verdict: {res_exp['prediction']} | Score: {res_exp['model_score']:.2%} | Version: {res_exp['model_info']['version']}")
    assert res_exp['prediction'] == "LIKELY REAL"
    assert res_exp['model_info']['version'] == "v3"

    # 3. Manipulated Face-Swap Video
    print("\n[3] Testing video_manipulated.mp4...")
    code_man, res_man = upload_video("video_manipulated.mp4", model="standard")
    print(f"    Standard     -> HTTP {code_man} | Verdict: {res_man['prediction']} | Score: {res_man['model_score']:.2%}")
    assert res_man['prediction'] == "LIKELY AI-GENERATED / MANIPULATED"

    # 4. Fully AI-Generated Video
    print("\n[4] Testing video_ai_generated.mp4...")
    code_ai, res_ai = upload_video("video_ai_generated.mp4", model="standard")
    print(f"    Standard     -> HTTP {code_ai} | Verdict: {res_ai['prediction']} | Score: {res_ai['model_score']:.2%}")
    assert res_ai['prediction'] != "LIKELY REAL", "CRITICAL ERROR: AI-generated video classified as REAL!"
    assert res_ai['prediction'] in ["LIKELY AI-GENERATED / MANIPULATED", "INCONCLUSIVE"]

    # 5. syn_30_known_ai_benchmark.mp4 (Conflict resolution)
    syn_file = "dataset_v3/synthetic/syn_30_known_ai_benchmark.mp4"
    print(f"\n[5] Testing syn_30_known_ai_benchmark.mp4...")
    code_k_std, res_k_std = upload_video(syn_file, model="standard")
    print(f"    Standard     -> HTTP {code_k_std} | Verdict: {res_k_std['prediction']} | Score: {res_k_std['model_score']:.2%}")

    code_k_exp, res_k_exp = upload_video(syn_file, model="experimental")
    print(f"    Experimental -> HTTP {code_k_exp} | Verdict: {res_k_exp['prediction']} | Score: {res_k_exp['model_score']:.2%}")
    print(f"    Status       : {res_k_exp['analysis_status']}")
    print(f"    Explanation  : {res_k_exp['explanation']}")
    assert res_k_exp['prediction'] != "LIKELY REAL", "CRITICAL REGRESSION: syn_30 produced REAL!"
    assert res_k_exp['prediction'] == "INCONCLUSIVE"
    assert "Conflicting Forensic Signals" in res_k_exp['analysis_status']

    # 6. Upload error: empty file
    print("\n[6] Testing 0-byte upload...")
    empty_path = "test_empty_tmp.mp4"
    with open(empty_path, "wb") as f:
        pass
    code_err, res_err = upload_video(empty_path, model="standard")
    print(f"    Empty File   -> HTTP {code_err} | Detail: {res_err.get('detail')}")
    assert code_err == 400
    if os.path.exists(empty_path):
        os.remove(empty_path)

    # 7. Upload error: unreadable / corrupt video
    print("\n[7] Testing corrupt video upload...")
    corrupt_path = "test_corrupt_tmp.mp4"
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPTED_STREAM_HEADER")
    code_cor, res_cor = upload_video(corrupt_path, model="experimental")
    print(f"    Corrupt Video -> HTTP {code_cor} | Verdict: {res_cor['prediction']} | Status: {res_cor.get('analysis_status')}")
    assert res_cor['prediction'] == "INCONCLUSIVE"
    assert "Failed" in res_cor['analysis_status'] or "Unreadable" in res_cor['analysis_status']
    if os.path.exists(corrupt_path):
        os.remove(corrupt_path)

    print("\n" + "=" * 70)
    print("ALL 7 LIVE SERVER TESTS PASSED 100%!")
    print("=" * 70)

if __name__ == "__main__":
    test_live_server()
