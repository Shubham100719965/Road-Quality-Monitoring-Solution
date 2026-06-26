import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Detection on a Single Image"
    )
    parser.add_argument(
        "--input",
        default="data/input/stop_sign.jpg",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--output",
        default="runs/detect/predict/processed_image.jpg",
        help="Path to save the annotated output image"
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Disable image display window"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = PROJECT_ROOT / "model" / "traffic_sign_detector.pt"
    image_path = Path(args.input)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Load trained YOLO model
    model = YOLO(str(model_path), task="detect")

    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    # Perform detection
    results = model(image)

    total_boxes = 0

    # Process detections
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            total_boxes += 1

    saved = cv2.imwrite(str(output_path), image)
    if saved:
        print(f"Saved output image: {output_path}")
    else:
        print(f"Error: Unable to save output image to {output_path}")

    if args.nogui:
        print(f"Processed: {image_path}")
        print(f"Detected Objects: {total_boxes}")
    else:
        cv2.imshow("Traffic Sign Detection", image)
        print(f"Detected Objects: {total_boxes}")
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
