import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Traffic sign detection on a video')
    parser.add_argument('--nogui', action='store_true', help='Disable video window display')
    parser.add_argument('--input', default='data/input/traffic_signs.mp4', help='Path to the input video file')
    parser.add_argument('--output', default='runs/detect/predict/processed_video.mp4', help='Path to save the annotated output video')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum number of frames to process')
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = PROJECT_ROOT / "model" / "traffic_sign_detector.pt"
    video_path = Path(args.input)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    detector = YOLO(str(model_path), task="detect")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Unable to open the input file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"Unable to create output video: {output_path}")
        cap.release()
        return

    frame_count = 0
    detections_total = 0
    while True:
        if args.max_frames is not None and frame_count >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        detections = detector(frame)
        for detection in detections:
            for bbox in detection.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detections_total += 1

        writer.write(frame)
        frame_count += 1
        if not args.nogui:
            cv2.imshow("Traffic sign detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if not args.nogui:
        cv2.destroyAllWindows()
    print(f"Saved output video: {output_path}")
    print(f"Processed {frame_count} frames and detected {detections_total} boxes.")


if __name__ == "__main__":
    main()