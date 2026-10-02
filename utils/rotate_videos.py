import cv2


def rotate_video_counterclockwise_sync(input_path: str, output_path: str):
    """Synchronous function to rotate a video 90 degrees using OpenCV."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # New width is original height
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # New height is original width
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError("Could not initialize video writer")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(rotated)


def rotate_video_clockwise_sync(input_path: str, output_path: str):
    """Synchronous function to rotate a video 90 degrees using OpenCV."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # New width is original height
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # New height is original width
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError("Could not initialize video writer")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        out.write(rotated)

    cap.release()
    out.release()