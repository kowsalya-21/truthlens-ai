import cv2, torch, numpy as np
from PIL import Image
from backend.detector import get_deepfake_model

frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def extract_face_crop_improved(frame_bgr):
    h, w = frame_bgr.shape[:2]
    aspect_ratio = w / float(h)
    is_square_crop = 0.9 <= aspect_ratio <= 1.1 and max(h, w) <= 512
    orientations = [(0, None), (90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    best_face = None
    best_rot_frame = None
    best_angle = 0
    best_method = None
    
    for angle, rot_code in orientations:
        rot_frame = frame_bgr if rot_code is None else cv2.rotate(frame_bgr, rot_code)
        rh, rw = rot_frame.shape[:2]
        gray = cv2.cvtColor(rot_frame, cv2.COLOR_BGR2GRAY)
        
        scale = 640.0 / max(rh, rw) if max(rh, rw) > 640 else 1.0
        small_gray = cv2.resize(gray, (int(rw * scale), int(rh * scale))) if scale != 1.0 else gray
        
        faces = frontal_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
        method = 'haar_frontal'
        
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
            method = 'haar_profile'
            
        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            best_face = (int(largest[0]/scale), int(largest[1]/scale), int(largest[2]/scale), int(largest[3]/scale))
            best_rot_frame = rot_frame
            best_angle = angle
            best_method = f'{method}_{angle}deg'
            break
            
    if best_face is not None:
        fx, fy, fw, fh = best_face
        rh, rw = best_rot_frame.shape[:2]
        side = int(max(fw, fh) * 1.25)
        cx, cy = fx + fw // 2, fy + fh // 2
        
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(rw, cx + side // 2)
        y2 = min(rh, cy + side // 2)
        
        crop = best_rot_frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb, True, {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1, 'method': best_method, 'angle': best_angle}
        
    if is_square_crop:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_ratio = float(np.sum(mask > 0) / (h * w))
        if skin_ratio >= 0.08:
            crop_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return crop_rgb, True, {'x': 0, 'y': 0, 'width': w, 'height': h, 'method': 'pre_cropped_square_face', 'angle': 0}
            
    return None, False, {'method': 'none'}

processor, model = get_deepfake_model()

for v in ['video_genuine.mp4', 'video_manipulated.mp4', 'video_ai_generated.mp4']:
    cap = cv2.VideoCapture(v)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * (total - 1) / 11) for i in range(12)]
    scores = []
    print(f"\n=== EVALUATING {v} ===")
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            crop_rgb, has_face, bbox = extract_face_crop_improved(frame)
            if has_face and crop_rgb is not None:
                inputs = processor(images=Image.fromarray(crop_rgb), return_tensors='pt')
                with torch.no_grad():
                    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
                p_fake = float(probs[1].item())
                scores.append(p_fake)
                m = bbox['method']
                print(f"  Frame {idx:3d}: method={m:24s} | fake_score={p_fake:.4f}")
            else:
                print(f"  Frame {idx:3d}: NO FACE DETECTED")
    if scores:
        mean_score = sum(scores)/len(scores)
        max_score = max(scores)
        min_score = min(scores)
        std_dev = (sum((s - mean_score)**2 for s in scores)/len(scores))**0.5
        print(f"Summary for {v}: mean={mean_score:.4f}, max={max_score:.4f}, min={min_score:.4f}, std={std_dev:.4f}, faces={len(scores)}/12")
