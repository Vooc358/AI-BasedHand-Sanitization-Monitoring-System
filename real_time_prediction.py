import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
from collections import defaultdict

# Create a folder to save the images
save_folder = "hand_frames"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 100
imgSize = 300
frame_count = 0
last_detection_time = time.time()

# Set the desired framerate
framerate = 5
delay_time = int(1000 / framerate)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    hands = detector.findHands(img, draw=False)  # Do not draw the skeleton

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop has valid dimensions
            # Save the cropped hand image
            frame_count += 1
            filename = os.path.join(save_folder, f"hand_frame_{frame_count}.jpg")
            cv2.imwrite(filename, imgCrop)
            print(f"Saved frame {frame_count}")

        last_detection_time = time.time()

    cv2.imshow("Image", img)

    # Check if no hand has been detected for 5 seconds
    if time.time() - last_detection_time > 5:
        print("No hand detected for 5 seconds. Closing video.")
        break

    if cv2.waitKey(delay_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#
# import os
# import numpy as np
# from ultralytics import YOLO
#
# model = YOLO('runs/classify/train2/weights/last.pt')  # load a custom model
#
# folder_path = 'hand_frames'  # Path to the folder containing images
#
# image_files = os.listdir(folder_path)
# image_paths = [os.path.join(folder_path, img_file) for img_file in image_files if
#                img_file.endswith(('.jpg', '.jpeg', '.png'))]
#
# class_counts = defaultdict(int)  # Dictionary to store counts of each class
#
# for image_path in image_paths:
#     results = model(image_path)  # predict on each image
#     names_dict = results[0].names
#     probs = results[0].probs.data.tolist()
#     print("Predictions for", image_path)
#     print(names_dict)
#     print(probs)
#
#     # Filter predictions with probability greater than 0.9
#     high_confidence_classes = [names_dict[i] for i, prob in enumerate(probs) if prob > 0.85]
#
#     if high_confidence_classes:
#         most_probable_class = max(set(high_confidence_classes), key=high_confidence_classes.count)
#         class_counts[most_probable_class] += 1
#         print("Most probable class:", most_probable_class)
#     else:
#         print("No class detected with probability greater than 0.5")
#
#     print("---------------------------")
#
# # Check if all class counts are more than twice
# sanitized = all(count > 2 for count in class_counts.values())
#
# # Print sanitization status
# if sanitized:
#     print("Sanitized properly")
# else:
#     print("Not sanitized")
#
# import shutil                   #deletes folder
# shutil.rmtree(save_folder)