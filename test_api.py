import urllib.request
import json
import os

def post_video(video_path):
    url = 'http://127.0.0.1:8000/api/analyze'
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
    
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode('utf-8'))

print('=== 1. TESTING REAL VIDEO API CALL ===')
res_real = post_video('video_genuine.mp4')
print('Prediction:', res_real['prediction'])
print('Model Fake Score:', res_real['model_score'])
print('Status:', res_real['analysis_status'])
print('Explanation:', res_real['explanation'])
print('Evidence items:', len(res_real['evidence']))

print('\n=== 2. TESTING MANIPULATED VIDEO API CALL ===')
res_fake = post_video('video_manipulated.mp4')
print('Prediction:', res_fake['prediction'])
print('Model Fake Score:', res_fake['model_score'])
print('Status:', res_fake['analysis_status'])
print('Explanation:', res_fake['explanation'])
print('Evidence items:', len(res_fake['evidence']))

print('\n=== 3. TESTING AI-GENERATED VIDEO API CALL ===')
res_ai = post_video('video_ai_generated.mp4')
print('Prediction:', res_ai['prediction'])
print('Model Fake Score:', res_ai['model_score'])
print('Status:', res_ai['analysis_status'])
print('Explanation:', res_ai['explanation'])
print('Frames Sampled:', res_ai['video_details']['frames_sampled'])
print('Frames with Faces:', f"{res_ai['video_details']['frames_with_faces']}/{res_ai['video_details']['frames_analyzed']}")
print('Faces Detected:', res_ai['video_details']['faces_detected'])
print('Faces Analyzed:', res_ai['video_details']['faces_analyzed'])
print('Evidence items:', len(res_ai['evidence']))
