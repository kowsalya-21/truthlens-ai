from backend.detector import analyze_video_file
from backend.experimental_detector import analyze_video_experimental

videos = [
    ("Known AI-Generated Video (known_ai_video.mp4)", "e:/truthlens-ai/known_ai_video.mp4"),
    ("Known Genuine Video (video_genuine.mp4)", "e:/truthlens-ai/video_genuine.mp4")
]

for title, path in videos:
    print("=" * 70)
    print(f"EVALUATING: {title}")
    print("=" * 70)
    
    # 1. Standard Mode
    std_res = analyze_video_file(path, max_samples=12)
    print("STANDARD MODE (Facial Specialist ViT):")
    print(f"  Prediction    : {std_res['prediction']}")
    print(f"  Model Score   : {std_res['model_score']}")
    print(f"  Status        : {std_res['analysis_status']}")
    print(f"  Explanation   : {std_res['explanation']}")
    print(f"  Faces Analyzed: {std_res['video_details']['faces_analyzed']}")
    print("-" * 50)
    
    # 2. Experimental Mode
    exp_res = analyze_video_experimental(path, num_frames=16)
    print("EXPERIMENTAL MODE (VideoMAE Spatiotemporal Head):")
    print(f"  Prediction    : {exp_res['prediction']}")
    print(f"  Model Score   : {exp_res['model_score']}")
    print(f"  Score Type    : {exp_res['metrics'].get('score_type')}")
    print(f"  Status        : {exp_res['analysis_status']}")
    print(f"  Explanation   : {exp_res['explanation']}")
    print(f"  Model Info    : {exp_res['model_info']['name']}")
    print(f"  Mode          : {exp_res['model_info']['mode']}")
    print()
