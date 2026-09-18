import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

images = []
labels = []

image_folder = "images"
annotation_folder = "annotations"

for xml_file in os.listdir(annotation_folder):

    tree = ET.parse(os.path.join(annotation_folder, xml_file))
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    for obj in root.findall("object"):

        label = obj.find("name").text

        if label == "with_mask":
            class_id = 1
        else:
            class_id = 0

        box = obj.find("bndbox")

        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)

        face = image[ymin:ymax, xmin:xmax]

        if face.size == 0:
            continue

        face = cv2.resize(face, (128,128))

        images.append(face)
        labels.append(class_id)

images = np.array(images) / 255.0
labels = to_categorical(labels,2)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128,activation="relu"),
    Dense(2,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=5,
    batch_size=32
)

model.save("mask_detector.h5")

print("Model saved as mask_detector.h5")