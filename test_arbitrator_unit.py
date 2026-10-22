import unittest
from backend.arbitrator import arbitrate_models

class TestArbitrator(unittest.TestCase):
    def test_unanimous_real(self):
        f_res = {
            "prediction": "LIKELY REAL",
            "model_score": 0.15,
            "video_details": {"faces_detected": 10, "faces_analyzed": 10},
            "metrics": {"persons_tracked": 1, "mean_fake_score": 0.15, "max_fake_score": 0.20},
            "explanation": "No facial manipulation."
        }
        v_raw = {
            "p_fake": 0.05,
            "p_real": 0.95,
            "tau_real": 0.25,
            "tau_fake": 0.70,
            "version": "v4"
        }
        res = arbitrate_models(f_res, v_raw)
        self.assertEqual(res["prediction"], "LIKELY REAL")
        self.assertTrue(res["arbitration_meta"]["agreement"])

    def test_unanimous_ai(self):
        f_res = {
            "prediction": "LIKELY AI-GENERATED / MANIPULATED",
            "model_score": 0.85,
            "video_details": {"faces_detected": 10, "faces_analyzed": 10},
            "metrics": {"persons_tracked": 1, "mean_fake_score": 0.85, "max_fake_score": 0.95},
            "explanation": "Face swap."
        }
        v_raw = {
            "p_fake": 0.92,
            "p_real": 0.08,
            "tau_real": 0.25,
            "tau_fake": 0.70,
            "version": "v4"
        }
        res = arbitrate_models(f_res, v_raw)
        self.assertEqual(res["prediction"], "LIKELY AI-GENERATED / MANIPULATED")
        self.assertTrue(res["arbitration_meta"]["agreement"])

    def test_conflict_facial_ai_scene_real(self):
        f_res = {
            "prediction": "LIKELY AI-GENERATED / MANIPULATED",
            "model_score": 0.75,
            "video_details": {"faces_detected": 10, "faces_analyzed": 10},
            "metrics": {"persons_tracked": 1, "mean_fake_score": 0.75, "max_fake_score": 0.85},
            "explanation": "Targeted face swap."
        }
        v_raw = {
            "p_fake": 0.10,
            "p_real": 0.90,
            "tau_real": 0.25,
            "tau_fake": 0.70,
            "version": "v4"
        }
        res = arbitrate_models(f_res, v_raw)
        self.assertEqual(res["prediction"], "INCONCLUSIVE")
        self.assertFalse(res["arbitration_meta"]["agreement"])
        self.assertIn("Conflicting Forensic Signals", res["analysis_status"])

    def test_conflict_facial_real_scene_ai(self):
        f_res = {
            "prediction": "LIKELY REAL",
            "model_score": 0.12,
            "video_details": {"faces_detected": 10, "faces_analyzed": 10},
            "metrics": {"persons_tracked": 1, "mean_fake_score": 0.12, "max_fake_score": 0.18},
            "explanation": "Faces look natural."
        }
        v_raw = {
            "p_fake": 0.88,
            "p_real": 0.12,
            "tau_real": 0.25,
            "tau_fake": 0.70,
            "version": "v4"
        }
        res = arbitrate_models(f_res, v_raw)
        self.assertEqual(res["prediction"], "INCONCLUSIVE")
        self.assertFalse(res["arbitration_meta"]["agreement"])
        self.assertIn("Conflicting Forensic Signals", res["analysis_status"])

    def test_no_faces_synthetic_animal(self):
        f_res = {
            "prediction": "INCONCLUSIVE",
            "model_score": None,
            "video_details": {"faces_detected": 0, "faces_analyzed": 0},
            "metrics": {"persons_tracked": 0},
            "explanation": "0 faces detected."
        }
        v_raw = {
            "p_fake": 0.99,
            "p_real": 0.01,
            "tau_real": 0.25,
            "tau_fake": 0.70,
            "version": "v4"
        }
        res = arbitrate_models(f_res, v_raw)
        self.assertEqual(res["prediction"], "LIKELY AI-GENERATED / MANIPULATED")
        self.assertIn("No human faces detected", res["explanation"])

    def test_no_faces_real_animation(self):
        f_res = {
            "prediction": "INCONCLUSIVE",
            "model_score": None,
            "video_details": {"faces_detected": 0, "faces_analyzed": 0},
            "metrics": {"persons_tracked": 0},
            "explanation": "0 faces detected."
        }
        v_raw = {
            "p_fake": 0.03,
            "p_real": 0.97,
            "tau_real": 0.25,
            "tau_fake": 0.70,
            "version": "v4"
        }
        res = arbitrate_models(f_res, v_raw)
        self.assertEqual(res["prediction"], "LIKELY REAL")
        self.assertIn("No human faces detected", res["explanation"])

if __name__ == "__main__":
    unittest.main()
