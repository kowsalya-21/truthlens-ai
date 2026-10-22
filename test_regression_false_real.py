import os
import unittest
from backend.detector import analyze_video_file
from backend.experimental_detector import analyze_video_experimental

class TestFalseRealRegression(unittest.TestCase):
    def setUp(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.ai_video_path = os.path.join(root_dir, "known_ai_video.mp4")
        self.genuine_video_path = os.path.join(root_dir, "video_genuine.mp4")
        self.assertTrue(os.path.exists(self.ai_video_path), "known_ai_video.mp4 must exist for regression test")
        self.assertTrue(os.path.exists(self.genuine_video_path), "video_genuine.mp4 must exist for regression test")

    def test_standard_detector_on_known_ai_video(self):
        """Standard detector must correctly identify manipulation or return inconclusive, never false-REAL."""
        res = analyze_video_file(self.ai_video_path, max_samples=12)
        print("\n[Standard Mode on known_ai_video]:", res["prediction"], f"(Score: {res['model_score']:.2%})")
        self.assertNotEqual(
            res["prediction"],
            "LIKELY REAL",
            "CRITICAL REGRESSION: Standard detector classified known AI video as REAL!"
        )
        self.assertEqual(res["prediction"], "LIKELY AI-GENERATED / MANIPULATED")

    def test_experimental_detector_on_known_ai_video(self):
        """Experimental detector must recognize cross-model conflict and return INCONCLUSIVE, never false-REAL."""
        res = analyze_video_experimental(self.ai_video_path, num_frames=16)
        print("\n[Experimental Mode on known_ai_video]:", res["prediction"], f"(Status: {res['analysis_status']}, Score: {res['model_score']:.2%})")
        self.assertNotEqual(
            res["prediction"],
            "LIKELY REAL",
            "CRITICAL REGRESSION: Experimental detector classified known AI video as REAL!"
        )
        self.assertEqual(res["prediction"], "INCONCLUSIVE")
        self.assertEqual(res["analysis_status"], "Inconclusive (Conflicting Forensic Signals)")

    def test_genuine_video_not_falsely_flagged(self):
        """Genuine video must remain LIKELY REAL in both modes."""
        std_res = analyze_video_file(self.genuine_video_path, max_samples=12)
        print("\n[Standard Mode on genuine video]:", std_res["prediction"], f"(Score: {std_res['model_score']:.2%})")
        self.assertEqual(
            std_res["prediction"],
            "LIKELY REAL",
            "Regression: Genuine video was falsely flagged as AI in standard mode!"
        )
        
        exp_res = analyze_video_experimental(self.genuine_video_path, num_frames=16)
        print("\n[Experimental Mode on genuine video]:", exp_res["prediction"], f"(Score: {exp_res['model_score']:.2%})")
        self.assertEqual(
            exp_res["prediction"],
            "LIKELY REAL",
            "Regression: Genuine video was falsely flagged as AI in experimental mode!"
        )

if __name__ == "__main__":
    unittest.main()
