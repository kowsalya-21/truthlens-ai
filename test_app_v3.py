import os
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_routes():
    print("=== Testing FastAPI App with TestClient ===")
    
    # 1. Health check
    h = client.get("/health").json()
    print(f"Health Version: {h['version']}")
    print(f"Experimental Model: {h['experimental_model']}")
    assert h["version"] == "3.0.0"
    assert "v3" in h["experimental_model"]
    
    # 2. Test analyze endpoint with model=experimental
    with open("video_genuine.mp4", "rb") as f:
        resp = client.post("/api/analyze?model=experimental", files={"file": ("video_genuine.mp4", f, "video/mp4")})
    assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"\n/api/analyze?model=experimental:")
    print(f"  Version: {data['model_info']['version']}")
    print(f"  Model Name: {data['model_info']['name']}")
    print(f"  Prediction: {data['prediction']}")
    print(f"  Score: {data['model_score']}")
    assert data["model_info"]["version"] == "v3"
    assert data["prediction"] == "LIKELY REAL"
    
    # 3. Test dedicated route /api/analyze/experimental?version=v3
    with open("video_genuine.mp4", "rb") as f:
        resp2 = client.post("/api/analyze/experimental?version=v3", files={"file": ("video_genuine.mp4", f, "video/mp4")})
    assert resp2.status_code == 200, f"Error {resp2.status_code}: {resp2.text}"
    data2 = resp2.json()
    print(f"\n/api/analyze/experimental?version=v3:")
    print(f"  Version: {data2['model_info']['version']}")
    print(f"  Prediction: {data2['prediction']}")
    assert data2["model_info"]["version"] == "v3"
    assert data2["prediction"] == "LIKELY REAL"

    # 4. Test dedicated route /api/analyze/experimental?version=v2
    with open("video_genuine.mp4", "rb") as f:
        resp3 = client.post("/api/analyze/experimental?version=v2", files={"file": ("video_genuine.mp4", f, "video/mp4")})
    assert resp3.status_code == 200, f"Error {resp3.status_code}: {resp3.text}"
    data3 = resp3.json()
    print(f"\n/api/analyze/experimental?version=v2:")
    print(f"  Version: {data3['model_info']['version']}")
    assert data3["model_info"]["version"] == "v2"

    print("\nALL FastAPI V3 Application Integration Routes Verified 100% Correct!")

if __name__ == "__main__":
    test_routes()
