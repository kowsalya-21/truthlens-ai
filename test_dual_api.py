import urllib.request
import json
import os

def post_video_to_endpoint(video_path, endpoint_url):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    filename = os.path.basename(video_path)
    
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    
    header = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: video/mp4\r\n\r\n'
    ).encode('utf-8')
    footer = f'\r\n--{boundary}--\r\n'.encode('utf-8')
    body = header + video_bytes + footer
    
    req = urllib.request.Request(endpoint_url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode('utf-8'))

videos = ['video_genuine.mp4', 'video_manipulated.mp4', 'video_ai_generated.mp4']

print("======================================================================")
print("1. TESTING PRODUCTION ENDPOINT: /api/analyze (Facial Specialist ViT)")
print("======================================================================")
for v in videos:
    res = post_video_to_endpoint(v, 'http://127.0.0.1:8000/api/analyze')
    print(f"Video     : {v}")
    print(f"Prediction: {res['prediction']}")
    print(f"Score     : {res['model_score']}")
    print(f"Status    : {res['analysis_status']}")
    print(f"Model Info: {res.get('model_info', {}).get('name')}")
    print("-" * 50)

print("\n======================================================================")
print("2. TESTING EXPERIMENTAL ENDPOINT: /api/analyze/experimental (VideoMAE)")
print("======================================================================")
for v in videos:
    res = post_video_to_endpoint(v, 'http://127.0.0.1:8000/api/analyze/experimental')
    print(f"Video     : {v}")
    print(f"Prediction: {res['prediction']}")
    print(f"Score     : {res['model_score']}")
    print(f"Status    : {res['analysis_status']}")
    print(f"Model Info: {res.get('model_info', {}).get('name')}")
    print("-" * 50)
