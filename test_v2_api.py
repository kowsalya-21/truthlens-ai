import urllib.request
import json
import os

def test_v2(endpoint):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    with open('video_ai_generated.mp4', 'rb') as f:
        data = f.read()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="video_ai_generated.mp4"\r\n'
        f'Content-Type: video/mp4\r\n\r\n'
    ).encode('utf-8') + data + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    req = urllib.request.Request(endpoint, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print(f"Endpoint: {endpoint}")
        print(f"  Prediction : {res['prediction']}")
        print(f"  Score      : {res['model_score']}")
        print(f"  Version    : {res['model_info'].get('version')}")
        print(f"  Model Name : {res['model_info'].get('name')}")
        print(f"  Status     : {res.get('analysis_status')}")

if __name__ == '__main__':
    test_v2('http://127.0.0.1:8000/api/analyze/experimental?version=v2')
    test_v2('http://127.0.0.1:8000/api/analyze?model=experimental_v2')
