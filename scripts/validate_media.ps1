# Vertex - Validate media file for biomechanics suitability
# Usage: .\validate_media.ps1 path\to\file.mp4

param([string]$File)

if (-not $File) { Write-Host "Usage: .\validate_media.ps1 <file>"; exit 1 }
if (-not (Test-Path $File)) { Write-Host "File not found: $File"; exit 1 }

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
& "$projectRoot\src\.venv\Scripts\Activate.ps1"

Push-Location $projectRoot
python -c "
import sys, cv2, mediapipe as mp, numpy as np
sys.path.insert(0, 'src')
from vertex.models import MODEL_PATH, L_SHOULDER, R_SHOULDER, R_INDEX, R_EAR, MOUTH_R

path = sys.argv[1]
is_video = path.lower().endswith(('.mp4','.avi','.mov','.mkv'))

opts = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)
lmk = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

if is_video:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps if fps > 0 else 0
    print(f'File:       {path}')
    print(f'Resolution: {w}x{h}  {fps:.0f}fps  {dur:.1f}s  ({total} frames)')
    sample_secs = [dur*0.1, dur*0.3, dur*0.5, dur*0.7]
    frames = []
    for s in sample_secs:
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
else:
    frame = cv2.imread(path)
    if frame is None: print('Cannot read image'); sys.exit(1)
    h, w = frame.shape[:2]
    print(f'File:       {path}')
    print(f'Resolution: {w}x{h}')
    frames = [frame]

detections, anchor_dists = 0, []
for frame in frames:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lmk.detect(mp_img)
    if not result.pose_landmarks: continue
    lms = result.pose_landmarks[0]
    detections += 1
    ls = np.array([lms[L_SHOULDER].x, lms[L_SHOULDER].y])
    rs = np.array([lms[R_SHOULDER].x, lms[R_SHOULDER].y])
    sw = np.linalg.norm(ls - rs)
    jaw = 0.4 * np.array([lms[R_EAR].x, lms[R_EAR].y]) + 0.6 * np.array([lms[MOUTH_R].x, lms[MOUTH_R].y])
    hand = np.array([lms[R_INDEX].x, lms[R_INDEX].y])
    ad = np.linalg.norm(hand - jaw) / max(sw, 0.01)
    anchor_dists.append(ad)
    r_wrist_vis = lms[16].visibility
    l_shldr_vis = lms[L_SHOULDER].visibility

lmk.close()

print()
print(f'Pose detected:   {detections}/{len(frames)} sampled frames')
if anchor_dists:
    ad_med = float(np.median(anchor_dists))
    print(f'Anchor dist (SW-normalised):  median={ad_med:.3f}')
    if ad_med > 1.5:
        print('  WARNING: anchor_dist > 1.5 SW -- camera is likely FRONT-FACING or too far oblique')
        print('  ACTION:  reposition camera to 90 deg SIDE PROFILE (perpendicular to shooting line)')
    elif ad_med > 0.8:
        print('  CAUTION: anchor_dist 0.8-1.5 SW -- camera is oblique, thresholds will be auto-calibrated')
    else:
        print('  OK: anchor_dist < 0.8 SW -- camera angle looks suitable for biomechanics')
    print()
    print('SUITABILITY CHECKLIST:')
    fps_ok = is_video and fps >= 25
    res_ok = w >= 1280
    print(f'  [{'OK' if fps_ok else 'FAIL'}] FPS >= 25 ({fps:.0f} fps)' if is_video else '  [N/A] FPS (image)')
    print(f'  [{'OK' if res_ok else 'WARN'}] Resolution >= 720p ({w}x{h})')
    print(f'  [{'OK' if detections > 0 else 'FAIL'}] Pose detected ({detections}/{len(frames)} frames)')
    print(f'  [{'OK' if ad_med <= 1.5 else 'FAIL'}] Camera angle (anchor_dist={ad_med:.2f} SW)')
" "$File" 2>&1 | Where-Object { $_ -notmatch "^W[0-9]|^INFO:|feedback|NORM_RECT|TensorFlow" }
Pop-Location