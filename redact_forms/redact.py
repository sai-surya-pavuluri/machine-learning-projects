import os
import time
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageHandler(FileSystemEventHandler):
    def __init__(self, input_folder, label_folder, output_folder):
        self.input_folder = input_folder
        self.label_folder = label_folder
        self.output_folder = output_folder

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png')):
            if 'redacted_' in os.path.basename(event.src_path):
                print(f"Skipping redacted image: {event.src_path}")
                return
            
            print(f"New image detected: {event.src_path}")
            self.process_image(event.src_path)

    def process_image(self, image_path):
        label_file_path = os.path.join(self.label_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        if not os.path.exists(label_file_path):
            print(f"Label file for {image_path} not found, skipping redaction.")
            return

        print(f"Label file found: {label_file_path}")

        with open(label_file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            print("Not enough detections in label file (need at least 2); skipping redaction.")
            return

        class_ids = []
        for line in lines:
            try:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    print(f"Skipping invalid label line: {line}")
                    continue
                class_id = int(parts[0])
                class_ids.append(class_id)
            except ValueError:
                print(f"Skipping invalid label line: {line}")
                continue

        if not any(cls in [2, 3] for cls in class_ids):
            print("No detection with class 2 or 3 found; skipping redaction.")
            return

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return

        height, width = image.shape[:2]

        for line in lines:
            try:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5:
                    continue
                cls, x_center, y_center, w, h = parts
                cls = int(cls)

                # Check if the coordinates are in pixel values or normalized
                if x_center > 1 or y_center > 1:  # Absolute pixel values
                    x_center = int(x_center)
                    y_center = int(y_center)
                    w = int(w)
                    h = int(h)
                else:  # Normalized coordinates
                    x_center = int(x_center * width)
                    y_center = int(y_center * height)
                    w = int(w * width)
                    h = int(h * height)

                if cls not in [0, 1]:
                    continue

                x1 = max(0, x_center - w // 2)
                y1 = max(0, y_center - h // 2)
                x2 = min(width, x_center + w // 2)
                y2 = min(height, y_center + h // 2)

                print(f"Redacting class {cls} box: ({x1}, {y1}) to ({x2}, {y2})")
                image[y1:y2, x1:x2] = (0, 0, 0)

            except Exception as e:
                print(f"Error processing line '{line}': {e}")
                continue

        redacted_image_path = os.path.join(self.output_folder, 'redacted_' + os.path.basename(image_path))
        cv2.imwrite(redacted_image_path, image)
        print(f"Redacted image saved: {redacted_image_path}")

def start_monitoring_redactions_folder(input_folder, label_folder, output_folder, stop_event):
    print("Started monitoring folder for redactions... ")
    event_handler = ImageHandler(input_folder, label_folder, output_folder)
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)
    observer.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    input_folder = 'runs/detect/results'  # Folder to monitor for new images
    label_folder = 'runs/detect/results/labels'  # Folder where the labels are saved
    output_folder = 'runs/detect/redact' # Folder to save redacted images

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    from threading import Event
    stop_event = Event()
    start_monitoring_redactions_folder(input_folder, label_folder, output_folder, stop_event)
