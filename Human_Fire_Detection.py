import numpy as np
import cv2
import os
import threading
import smtplib

# Paths to cascade models
face_cascade_path = "/home/ashutosh007/Desktop/proj/haarcascade_frontalface_alt.xml"
fire_cascade_path = "/home/ashutosh007/Desktop/proj/fire_detection_cascade_model.xml"

# Initialize cascades
face_cascade = cv2.CascadeClassifier(face_cascade_path)
fire_cascade = cv2.CascadeClassifier(fire_cascade_path)

if face_cascade.empty() or fire_cascade.empty():
    print("Error: One or more cascade classifiers not loaded. Please check the file paths.")
    exit()

# KNN Helper Functions for Face Recognition
def preprocess_image(img):
    return img / 255.0  # Scale pixel values between 0 and 1

def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []
    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))

    dlist = sorted(dlist, key=lambda x: x[0])
    top_k_labels = [dlist[i][1] for i in range(k)]
    labels, cnts = np.unique(top_k_labels, return_counts=True)
    idx = cnts.argmax()
    return int(labels[idx])

# Load face recognition dataset
dataset_path = "/home/ashutosh007/Desktop/proj/data/"
faceData, labels = [], []
nameMap = {}
classId = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]
        dataItem = np.load(os.path.join(dataset_path, f))
        m = dataItem.shape[0]
        faceData.append(dataItem)
        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels, axis=0).reshape((-1, 1))
XT = preprocess_image(XT)

# Function to send email after fire detection
def send_mail_function():
    recipientmail = "recipient@example.com"
    sender_email = "sender@example.com"
    sender_password = "your_app_password"

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        subject = "Fire Alert"
        body = "Warning: A fire has been detected!"
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(sender_email, recipientmail, message)
        print(f"Alert email sent successfully to {recipientmail}")
        server.close()
    except Exception as e:
        print(f"Error while sending email: {e}")

# Variable to ensure fire notification is sent only once
fire_email_sent = False

# Start video capture
vid = cv2.VideoCapture(0)
offset = 20  # Offset for cropping faces

while True:
    ret, frame = vid.read()
    if not ret:
        print("Error accessing camera feed.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Detection and Recognition
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        if x - offset < 0 or y - offset < 0 or x + w + offset > frame.shape[1] or y + h + offset > frame.shape[0]:
            continue

        cropped_face = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        if cropped_face.size > 0:
            cropped_face = cv2.resize(cropped_face, (100, 100))
            cropped_face = preprocess_image(cropped_face.flatten())
            classPredicted = knn(XT, yT, cropped_face)
            namePredicted = nameMap[classPredicted]

            cv2.putText(frame, namePredicted, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Fire Detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fires = fire_cascade.detectMultiScale(gray_frame, 1.2, 5)
        # Define red color range in HSV
    lower_red1 = (0, 50, 50)    # Lower range for red (e.g., pure red)
    upper_red1 = (10, 255, 255) # Upper range for red
    lower_red2 = (170, 50, 50)  # Another range for red (wraparound)
    upper_red2 = (180, 255, 255)
    for (x, y, w, h) in fires:
        # Extract the region of interest (ROI)
        roi_hsv = hsv[y:y+h, x:x+w]
         # Create masks for red color
        mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate the percentage of red pixels in the ROI
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
        red_ratio = red_pixels / total_pixels
           # Intensity level based on red_ratio
        fire_intensity = int(red_ratio * 100)  # Convert to percentage
    
    # Display the intensity level
        intensity_text = f"Intensity: {fire_intensity}%"
    
        
        # Threshold for red intensity (e.g., 20% of ROI should be red)
        if red_ratio > 0.2:  # Adjust threshold as needed
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 255), 2)
            cv2.putText(frame, 'Fire Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, intensity_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if not fire_email_sent:
                print("Fire detected! Sending email...")
                threading.Thread(target=send_mail_function).start()
                fire_email_sent = True

    # Show the combined output
    cv2.imshow("face and Fire Detection", frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()