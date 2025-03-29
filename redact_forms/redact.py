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
        # Check if it's a new image file
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
            if 'redacted_' in os.path.basename(event.src_path):
                print(f"Skipping redacted image: {event.src_path}")
                return
            
            print(f"New image detected: {event.src_path}")
            self.process_image(event.src_path)

    def process_image(self, image_path):
        # Get the name of the label file associated with the image
        label_file_path = os.path.join(self.label_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')

        if os.path.exists(label_file_path):  # Ensure we check the correct path
            print(f"Label file found: {label_file_path}")
            
            # Example: Reading the label file and extracting bounding box coordinates to redact
            with open(label_file_path, 'r') as f:
                lines = f.readlines()

            # Read image
            image = cv2.imread(image_path)

            for line in lines:
                # Unpack the 5 values: class_id, x_center, y_center, width, height
                try:
                    _, x_center, y_center, width, height = map(float, line.strip().split())

                    # Convert normalized coordinates to pixel values
                    x_center = int(x_center * image.shape[1])  # Convert to pixel values
                    y_center = int(y_center * image.shape[0])
                    width = int(width * image.shape[1])
                    height = int(height * image.shape[0])

                    # Calculate bounding box coordinates
                    x1 = x_center - width // 2
                    y1 = y_center - height // 2
                    x2 = x_center + width // 2
                    y2 = y_center + height // 2

                    # Redact the region of interest (e.g., black out the area)
                    image[y1:y2, x1:x2] = (0, 0, 0)  # Example: Black out the area
                except ValueError:
                    print(f"Skipping invalid label line: {line}")
                    continue  # Skip invalid label lines

            # Save the redacted image
            redacted_image_path = os.path.join(self.output_folder, 'redacted_' + os.path.basename(image_path))
            cv2.imwrite(redacted_image_path, image)
            print(f"Redacted image saved: {redacted_image_path}")
        else:
            print(f"Label file for {image_path} not found, skipping redaction.")

def start_monitoring(input_folder, label_folder, output_folder):
    print("Started monitoring folder: ")
    event_handler = ImageHandler(input_folder, label_folder, output_folder)
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    input_folder = 'runs/detect/results'  # Folder to monitor for new images
    label_folder = 'runs/detect/results/labels'  # Folder where the labels are saved
    output_folder = 'runs/detect/redact' # Folder to save redacted images

    # Create the results folder if it doesn't exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start monitoring the folder
    start_monitoring(input_folder, label_folder, output_folder)
