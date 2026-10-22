import requests
import json
import io

def test_corrupt_file_handling():
    url = 'http://127.0.0.1:8000/api/analyze/experimental'
    corrupt_data = io.BytesIO(b"Not a valid MP4 header or video stream bytes sequence 1234567890")
    files = {'file': ('corrupt.mp4', corrupt_data, 'video/mp4')}
    
    response = requests.post(url, files=files, timeout=15)
    print("HTTP Status Code:", response.status_code)
    try:
        data = response.json()
        print("Response JSON:\n", json.dumps(data, indent=2))
        
        # Verify failure is cleanly reported and NOT substituted with standard detector
        assert data.get("status") == "error", f"Expected status 'error', got {data.get('status')}"
        assert "Unreadable Video" in data.get("analysis_status", ""), f"Unexpected analysis_status: {data.get('analysis_status')}"
        assert data.get("model_info", {}).get("mode") == "experimental", f"Unexpected mode: {data.get('model_info')}"
        # Confirm facial detector fields are absent or null
        assert data.get("video_details", {}).get("faces_detected") is None, "Standard facial detector output leaked into experimental error!"
        print("\n>>> CONFIRMED: Experimental mode failure reported cleanly and NO standard-mode substitution occurred.")
    except Exception as e:
        print("Verification failed:", e)
        raise

if __name__ == "__main__":
    test_corrupt_file_handling()
