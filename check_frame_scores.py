from backend.detector import analyze_video_file

res = analyze_video_file('known_ai_video.mp4', max_samples=12)
print("Prediction:", res["prediction"])
print("Model Score:", res["model_score"])
for f in res["frame_analysis"]:
    print(f"Frame {f['frame_index']:3d} (ts={f['timestamp_sec']:5.2f}s) | Detected: {str(f['face_detected']):5s} | Fake: {f['fake_score']}")
