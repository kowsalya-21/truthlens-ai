import urllib.request
import json
import os

def test_api_route(endpoint):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    with open('video_genuine.mp4', 'rb') as f:
        data = f.read()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="video_genuine.mp4"\r\n'
        f'Content-Type: video/mp4\r\n\r\n'
    ).encode('utf-8') + data + f'\r\n--{boundary}--\r\n'.encode('utf-8')
    req = urllib.request.Request(endpoint, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print(f"\nEndpoint: {endpoint}")
        print(f"  Prediction : {res['prediction']}")
        print(f"  Score      : {res['model_score']}")
        print(f"  Version    : {res['model_info'].get('version')}")
        print(f"  Model Name : {res['model_info'].get('name')}")
        print(f"  Status     : {res.get('analysis_status')}")
        return res

if __name__ == '__main__':
    # Test v3 dedicated endpoint
    res1 = test_api_route('http://127.0.0.1:8001/api/analyze/experimental?version=v3')
    assert res1['model_info']['version'] == 'v3', f"Expected v3, got {res1['model_info']['version']}"
    assert res1['prediction'] == 'LIKELY REAL', f"Expected LIKELY REAL, got {res1['prediction']}"
    
    # Test v3 via default experimental parameter
    res2 = test_api_route('http://127.0.0.1:8001/api/analyze?model=experimental')
    assert res2['model_info']['version'] == 'v3', f"Expected v3, got {res2['model_info']['version']}"
    assert res2['prediction'] == 'LIKELY REAL', f"Expected LIKELY REAL, got {res2['prediction']}"
    
    print("\nAPI Integration Tests Passed Successfully!")
