from backend.detector import analyze_video_file
from backend.experimental_detector import analyze_video_experimental

genuine_path = "e:/truthlens-ai/video_genuine.mp4"

print("======================================================================")
print("TESTING KNOWN GENUINE VIDEO: video_genuine.mp4")
print("======================================================================")
std_gen = analyze_video_file(genuine_path, max_samples=12)
print("STANDARD DETECTOR:")
print(f"  Prediction : {std_gen['prediction']}")
print(f"  Fake Score : {std_gen['model_score']:.4f}")
print(f"  Status     : {std_gen['analysis_status']}")
print(f"  Explanation: {std_gen['explanation']}")
print("-" * 50)

exp_gen = analyze_video_experimental(genuine_path, num_frames=16)
print("EXPERIMENTAL DETECTOR:")
print(f"  Prediction : {exp_gen['prediction']}")
print(f"  Fake Score : {exp_gen['model_score']:.4f}")
print(f"  Status     : {exp_gen['analysis_status']}")
print(f"  Explanation: {exp_gen['explanation']}")
print("======================================================================")
