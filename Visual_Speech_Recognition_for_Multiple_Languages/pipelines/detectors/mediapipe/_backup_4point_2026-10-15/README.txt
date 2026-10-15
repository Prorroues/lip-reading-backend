Backup of the live MediaPipe 4-keypoint crop (2026-10-15).

What this was:
- detector.py: MediaPipe FaceDetection, 4 points (eyes / nose / mouth center)
- video_process.py: crop a 96x96 box around the mouth-center point (start_idx=3, stop_idx=4)

Restore:
  copy detector.py and video_process.py one level up, overwriting the live files.
  Then restart the VSR backend.
