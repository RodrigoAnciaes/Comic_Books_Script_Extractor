# Imports
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import numpy as np

# Directories
input_dir = Path('./generate_input')
input_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path('./generate_output_balloons')
output_dir.mkdir(parents=True, exist_ok=True)

overlay_dir = output_dir / 'overlays'
overlay_dir.mkdir(parents=True, exist_ok=True)
overlay_prefix = "" # Image prefix, can be empty
overlay_suffix = "" # Image suffix, can be empty

detection_dir = output_dir / 'detections'
detection_dir.mkdir(parents=True, exist_ok=True)
detection_prefix = ""  # Text prefix, can be empty
detection_suffix = ""  # Text suffix, can be empty

mask_dir = output_dir / 'masks'
mask_dir.mkdir(parents=True, exist_ok=True)
mask_prefix = ""  # Text prefix, can be empty
mask_suffix = ""  # Text suffix, can be empty

# Load your trained model
model_path = 'training_output_balloons/watermark5/weights/best.pt' # THe model returned from the training script
model = YOLO(model_path)

# Mode selection: detection or segmentation
mode = "detection"

# Split multiple detections or keep them together?
# Todo

# Classes to detect
# Example: ['SpeechBalloons', 'General_speech', 'hit_sound', 'blast_sound', 'narration speech', 'thought_speech', 'roar']
selected_classes = ['SpeechBalloons', 'General_speech', 'hit_sound', 'blast_sound', 'narration speech', 'thought_speech', 'roar']

# Class override mapping, treats the left side of the mapping as if it was the class of the right side
# Example: thought_speech annotations will be treated as SpeechBalloons annotations.
class_overrides = {
    
}

# Confidence threshold
confidence_threshold = 0.15

# Label settings
label_boxes = True  # Draw class names or just boxes
font_size = 30  # Font size for the class labels

try:
    font = ImageFont.truetype("arial.ttf", 30)  # Update font size as needed
except IOError:
    font = ImageFont.load_default()
    print("Default font will be used, as custom font not found.")

# Label colors by index
predefined_colors_with_text = [
    ((204, 0, 0),     'white'),  # Darker red, white text
    ((0, 204, 0),     'black'),  # Darker green, black text
    ((0, 0, 204),     'white'),  # Darker blue, white text
    ((204, 204, 0),   'black'),  # Darker yellow, black text
    ((204, 0, 204),   'white'),  # Darker magenta, white text
    ((0, 204, 204),   'black'),  # Darker cyan, black text
    ((153, 0, 0),     'white'),  # Darker maroon, white text
    ((0, 153, 0),     'white'),  # Darker green, white text
    ((0, 0, 153),     'white'),  # Darker navy, white text
    ((153, 153, 0),   'black'),  # Darker olive, black text
    # Add more color pairs if needed
]

# Assign colors to each class
class_colors = {class_name: predefined_colors_with_text[i][0] for i, class_name in enumerate(selected_classes)}
text_colors = {class_name: predefined_colors_with_text[i][1] for i, class_name in enumerate(selected_classes)}

# Store input images in a variable
image_paths = []
for extension in ['*.jpg', '*.jpeg', '*.png']:
    image_paths.extend(input_dir.glob(extension))

# Segmentation class
class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, _ = img.shape
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]

        segmentation_contours_idx = []
        if len(result) > 0:
            for seg in result.masks.xy:
                segment = np.array(seg, dtype=np.float32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores

ys = YOLOSEG(model_path)

# Function to estimate text size
def estimate_text_size(label, font_size):
    approx_char_width = font_size * 0.6
    text_width = len(label) * approx_char_width
    text_height = font_size
    return text_width, text_height

def write_detections_to_file(image_path, detections):
    # Create a text file named after the image
    text_file_path = detection_dir / f"{detection_prefix}{image_path.stem}{detection_suffix}.txt"

    with open(text_file_path, 'w') as file:
        for detection in detections:
            file.write(f"{detection}\n")


def generate_batch_balloons():
    # Process images with progress bar
    print(f"Generating outputs in {mode} mode.")
    for image_path in tqdm(image_paths, desc='Processing Images'):
        # Detection Mode
        if mode == "detection":
            img_cv = cv2.imread(str(image_path))  # Load the image with OpenCV for mask generation
            mask_img = np.zeros(img_cv.shape[:2], dtype=np.uint8)  # Initialize a blank mask for all detections

            img_pil = Image.open(image_path)  # Load the image with PIL for overlay generation
            results = model.predict(img_pil)
            draw = ImageDraw.Draw(img_pil)
            detections = []

            if len(results) > 0 and results[0].boxes.xyxy is not None:
                for idx, box in enumerate(results[0].boxes.xyxy):
                    x1, y1, x2, y2 = box[:4].tolist()
                    cls_id = int(results[0].boxes.cls[idx].item())
                    conf = results[0].boxes.conf[idx].item()
                    cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
                    cls_name = class_overrides.get(cls_name, cls_name)
                    #print(f"Detected {cls_name} with confidence {conf:.2f}")
                    if cls_name in selected_classes and conf >= confidence_threshold:

                        if cls_name == 'narration speech':
                            cls_name = 'narration_speech'

                        # save each os the balloons to a separate image
                        # crop the image
                        ballon_img = img_pil.crop((x1, y1, x2, y2))
                        ballon_path = output_dir / 'speech_balloons' / f"{image_path.stem}_{cls_name}_{conf:.2f}.png"
                        ballon_img.save(ballon_path)
                        # Save the coordinates of the balloon
                        with open(output_dir / 'speech_balloons_coordinates' / f"{image_path.stem}_{cls_name}_{conf:.2f}.txt", 'w') as f:
                            f.write(f"{x1} {y1} {x2} {y2}")

                        box_color = class_colors.get(cls_name, (255, 0, 0))
                        text_color = text_colors.get(cls_name, 'black')
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)

                        # Fill mask image for this detection
                        cv2.rectangle(mask_img, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)  # -1 thickness fills the rectangle

                        if label_boxes:
                            label = f"{cls_name}: {conf:.2f}"
                            text_size = estimate_text_size(label, font_size)
                            draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=box_color)
                            draw.text((x1, y1 - text_size[1] - 5), label, fill=text_color, font=font)

                        # Add detection data to the list
                        detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")


                        

            # Save overlay images
            img_pil.save(overlay_dir / f"{overlay_prefix}{image_path.stem}{overlay_suffix}{image_path.suffix}")

            # Write detections to a text file
            write_detections_to_file(image_path, detections)

            # Save the combined mask image
            mask_output_path = mask_dir / f"{mask_prefix}{image_path.stem}{mask_suffix}.png"
            cv2.imwrite(str(mask_output_path), mask_img)
            


    print(f"Processed {len(image_paths)} images. Overlays saved to '{overlay_dir}', Detections saved to '{detection_dir}', and Masks saved to '{mask_dir}',and Speech balloons saved to 'speech_balloons' folder.")


#generate_batch()


def generate_unit_balloons(image):
    """Receives comic image and generates speech balloons for a single image. 
    Outputs created in generate_output_balloons_unit directory"""

    output_dir = Path('./generate_output_balloons_unit')
    output_dir.mkdir(parents=True, exist_ok=True)
    img_pil = Image.open(image)  # Load the image with PIL for overlay generation
    original_img = img_pil.copy()
    results = model.predict(img_pil)
    draw = ImageDraw.Draw(img_pil)
    detections = []
    count_img = 0

    if len(results) > 0 and results[0].boxes.xyxy is not None:
        for idx, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = box[:4].tolist()
            cls_id = int(results[0].boxes.cls[idx].item())
            conf = results[0].boxes.conf[idx].item() # Confidence
            cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
            cls_name = class_overrides.get(cls_name, cls_name)
            #print(f"Detected {cls_name} with confidence {conf:.2f}")
            if cls_name in selected_classes and conf >= confidence_threshold:

                if cls_name == 'narration speech':
                    cls_name = 'narration_speech'
                # save each os the balloons to a separate image
                # crop the image
                ballon_img = original_img.crop((x1, y1, x2, y2))
                image_name = f"{image.stem}_{cls_name}_{conf:.2f}.png"
                #check if the image already exists
                if (output_dir / 'speech_balloons' / image_name).exists():
                    # change the name of the image
                    count_img += 1
                    image_name = f"{image.stem}_{cls_name}_{conf:.2f}_{count_img}.png"
                ballon_path = output_dir / 'speech_balloons' / image_name
                # create the directory if it does not exist
                ballon_path.parent.mkdir(parents=True, exist_ok=True)
                ballon_img.save(ballon_path)
                # Save the coordinates of the balloon
                coodinates_path = output_dir / 'speech_balloons_coordinates' / f"{image.stem}_{cls_name}_{conf:.2f}.txt"
                coodinates_path.parent.mkdir(parents=True, exist_ok=True)
                with open(coodinates_path, 'w') as f:
                    f.write(f"{x1} {y1} {x2} {y2}")

                box_color = class_colors.get(cls_name, (255, 0, 0))
                text_color = text_colors.get(cls_name, 'black')
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=7)

                mask_img = np.zeros(img_pil.size, dtype=np.uint8)

                # Fill mask image for this detection
                cv2.rectangle(mask_img, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)  # -1 thickness fills the rectangle

                if label_boxes:
                    label = f"{cls_name}: {conf:.2f}"
                    text_size = estimate_text_size(label, font_size)
                    draw.rectangle([x1, y1 - text_size[1] - 5, x1 + text_size[0], y1], fill=box_color)
                    draw.text((x1, y1 - text_size[1] - 5), label, fill=text_color, font=font)

                # Add detection data to the list
                detections.append(f"{cls_name} {conf:.2f} {x1} {y1} {x2} {y2}")

    overlay_dir = output_dir / 'overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    # Save overlay images
    img_pil.save(overlay_dir / f"{overlay_prefix}{image.stem}{overlay_suffix}{image.suffix}")

    # Write detections to a text file
    detection_dir = output_dir / 'detections'
    detection_dir.mkdir(parents=True, exist_ok=True)
    text_file_path = detection_dir / f"{detection_prefix}{image.stem}{detection_suffix}.txt"
    with open(text_file_path, 'w') as file:
        for detection in detections:
            file.write(f"{detection}\n")
    
    mask_dir = output_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Save the combined mask image
    mask_output_path = mask_dir / f"{mask_prefix}{image.stem}{mask_suffix}.png"
    cv2.imwrite(str(mask_output_path), mask_img)

    # Save the probabilities
    class_prob_path = output_dir / 'class_probabilities' / f"{image.stem}_probabilities.txt"
    class_prob_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(class_prob_path, 'w') as f:
        for idx, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = box[:4].tolist()
            cls_id = int(results[0].boxes.cls[idx].item())
            conf = results[0].boxes.conf[idx].item()
            cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
            cls_name = class_overrides.get(cls_name, cls_name)
            f.write(f"{cls_name} {conf:.2f}\n")
    print(f"Processed {image}. Overlays saved to '{overlay_dir}', Detections saved to '{detection_dir}', and Masks saved to '{mask_dir}',and Speech balloons saved to 'speech_balloons' folder.")


    print(f"Processed {len(image_paths)} images. Overlays saved to '{overlay_dir}', Detections saved to '{detection_dir}', and Masks saved to '{mask_dir}',and Speech balloons saved to 'speech_balloons' folder.")


def generate_prob_unity_balloons(image):
    """Receives comic image and returns a list with the probabilities of each balloon class detected in the image."""
    img_pil = Image.open(image)  # Load the image with PIL for overlay generation
    results = model.predict(img_pil)
    # print the probabilities of each class]
    ret = []	
    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = box[:4].tolist()
        cls_id = int(results[0].boxes.cls[idx].item())
        conf = results[0].boxes.conf[idx].item()
        cls_name = results[0].names[cls_id] if 0 <= cls_id < len(results[0].names) else "Unknown"
        cls_name = class_overrides.get(cls_name, cls_name)
        ret.append((cls_name,conf))

    return ret