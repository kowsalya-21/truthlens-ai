import json
from backend.detector import analyze_video_file

videos = ['video_genuine.mp4', 'video_manipulated.mp4', 'video_ai_generated.mp4']
for v in videos:
    print(f"\n==========================================")
    print(f"EVALUATING: {v}")
    print(f"==========================================")
    res = analyze_video_file(v, max_samples=12)
    print(f"Prediction: {res['prediction']}")
    print(f"Model Score (fake probability): {res['model_score']}")
    print(f"Analysis Status: {res['analysis_status']}")
    print(f"Explanation: {res['explanation']}")
    print(f"Metrics: {json.dumps(res['metrics'], indent=2)}")
    v_det = res['video_details']
    print(f"Frames Sampled: {v_det['frames_sampled']}")
    print(f"Frames With Faces: {v_det['frames_with_faces']}/{v_det['frames_analyzed']} ({(v_det['frames_with_faces']/max(1, v_det['frames_analyzed'])):.1%})")
    print(f"Faces Detected: {v_det['faces_detected']}")
    print(f"Faces Analyzed: {v_det['faces_analyzed']}")
    print("Frame details:")
    for f in res['frame_analysis']:
        idx = f['frame_index']
        t = f['timestamp_sec']
        fd = f['face_detected']
        score = f['fake_score']
        method = f['bbox'].get('method')
        bbox = f['bbox']
        print(f"  Frame {idx:3d} (t={t:5.2f}s) | face={fd} | fake_score={score} | method={method} | bbox={bbox}")
