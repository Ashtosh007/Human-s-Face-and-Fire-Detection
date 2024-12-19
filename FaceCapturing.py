import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog

# Function to get user input via GUI
def get_dataset_name():
    root = tk.Tk()
    root.withdraw()  # Hiding the main tkinter window
    dataset_name = simpledialog.askstring("Dataset Name", "Enter your name for the dataset:")
    root.destroy()  # Closing the tkinter instance
    return dataset_name

# Initialize camera
cam = cv2.VideoCapture(0)

#  the dataset name from the user
fileName = get_dataset_name()

if not fileName:
    print("Dataset name not provided. Exiting.")
    cam.release()
    cv2.destroyAllWindows()
    exit()

dataset_path = "/home/ashutosh007/Desktop/proj/data/"
offset = 20

# Load the Haar Cascade model
model = cv2.CascadeClassifier("/home/ashutosh007/Desktop/proj/haarcascade_frontalface_alt.xml")

faceData = []
skip = 0

while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")
        break  # Exit the loop if camera reading fails

    # Convert the image to grayscale for face detection
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = model.detectMultiScale(grayImg, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    if len(faces) > 0:
        f = faces[-1]
        x, y, w, h = f
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y - offset:y + h + offset, x - offset:x + w + offset]
        cropped_face = cv2.resize(cropped_face, (100, 100))
        skip += 1

        if skip % 10 == 0:
            faceData.append(cropped_face)
            print("Saved so far: " + str(len(faceData)))

    # Show the image with detections
    cv2.imshow("Image Window", img)

    # Check for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close windows
faceData = np.asarray(faceData)
m = faceData.shape[0]
faceData = faceData.reshape((m, -1))

print(faceData.shape)

filepath = dataset_path + fileName + ".npy"
np.save(filepath, faceData)
print("Data Saved Successfully at: " + filepath)

cam.release()
cv2.destroyAllWindows()
