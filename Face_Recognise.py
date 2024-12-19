import cv2
import numpy as np
import os

# Helper function to scale/normalize the image
def preprocess_image(img):
    return img / 255.0  # Scale pixel values between 0 and 1

def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

def knn(X, y, xt, k=5):
    m = X.shape[0]
    dlist = []
    for i in range(m):
        d = dist(X[i], xt)
        dlist.append((d, y[i]))  # Append distance and corresponding label

    # Sort dlist by distance (the first element of each tuple)
    dlist = sorted(dlist, key=lambda x: x[0])  # Sort by distance

    # Extract the labels of the top k neighbors
    top_k_labels = [dlist[i][1] for i in range(k)]  # Extract labels only

    # Find the most frequent label
    labels, cnts = np.unique(top_k_labels, return_counts=True)
    idx = cnts.argmax()  # Get the index of the label with max occurrences
    pred = labels[idx]

    return int(pred)

# Load your data and train the KNN
dataset_path = "/home/ashutosh007/Desktop/proj/data/"
faceData = []
offset = 20
labels = []
nameMap = {}
classId = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        nameMap[classId] = f[:-4]
        # Load the dataset
        dataItem = np.load(os.path.join(dataset_path, f))  # Correct path
        m = dataItem.shape[0]
        faceData.append(dataItem)

        target = classId * np.ones((m,))
        classId += 1
        labels.append(target)

if len(faceData) == 0 or len(labels) == 0:
    print("Face data not found in the dataset.")
    exit()
XT = np.concatenate(faceData, axis=0)
yT = np.concatenate(labels, axis=0).reshape((-1, 1))

# Preprocess the data
XT = preprocess_image(XT)  # Scale the dataset

print(XT.shape)
print(yT.shape)
print(nameMap)

# Prediction loop~
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("/home/ashutosh007/Desktop/proj/haarcascade_frontalface_alt.xml")

while True:
    success, img = cam.read()
    if not success:
        print("Reading Camera Failed!")
        break

    # Convert the image to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = model.detectMultiScale(grayImg, 1.3, 5)

    for f in faces:
        x, y, w, h = f
        
        # Boundary check: Ensure coordinates stay within image dimensions
        if x - offset < 0 or y - offset < 0 or x + w + offset > img.shape[1] or y + h + offset > img.shape[0]:
            continue  # Skip if face goes out of bounds
        
        # Crop the face
        cropped_face = img[y - offset:y+h + offset, x - offset:x+w + offset]

        # If cropped_face is valid, resize it
        if cropped_face.size > 0:  # Check if it's not empty
            cropped_face = cv2.resize(cropped_face, (100, 100))

            # Preprocess the cropped face
            cropped_face = preprocess_image(cropped_face.flatten())  # Scale input

            # Predict using KNN
            classPredicted = knn(XT, yT, cropped_face)
            namePredicted = nameMap[classPredicted]

            # Display prediction on the image
            cv2.putText(img, namePredicted, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Prediction window", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
