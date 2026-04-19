import os

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# ---------------------------- #
#  1️⃣ CONVERT FER2013 CSV TO IMAGES
# ---------------------------- #
csv_file = r"D:\moodify\fer2013.csv"
output_dir = r"D:\moodify\fer2013_images"

emotion_labels = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
    4: "Neutral", 5: "Sad", 6: "Surprise"
}

df = pd.read_csv(csv_file)
usage_map = {"Training": "train", "PublicTest": "validation", "PrivateTest": "test"}

for usage in ["train", "validation", "test"]:
    for emotion in emotion_labels.values():
        os.makedirs(os.path.join(output_dir, usage, emotion), exist_ok=True)

for index, row in df.iterrows():
    emotion = emotion_labels[row["emotion"]]
    pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    usage = usage_map.get(row["Usage"], "train")
    image_path = os.path.join(output_dir, usage, emotion, f"{index}.jpg")
    cv2.imwrite(image_path, pixels)

print("✅ Image conversion complete!")

# ---------------------------- #
#  2️⃣ LOAD DATASET & TRAIN MODEL
# ---------------------------- #
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 50

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    output_dir + "/train",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    output_dir + "/validation",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# ---------------------------- #
#  3️⃣ LOAD PRETRAINED MODEL & MODIFY
# ---------------------------- #
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=AdamW(learning_rate=0.0001), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
              metrics=['accuracy'])

print("✅ Model built successfully!")

# ---------------------------- #
#  4️⃣ TRAIN MODEL
# ---------------------------- #
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# ---------------------------- #
#  5️⃣ FINE-TUNE: UNFREEZE LAST 100 LAYERS
# ---------------------------- #
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(optimizer=AdamW(learning_rate=0.00003), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), 
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# ---------------------------- #
#  6️⃣ EVALUATE & SAVE MODEL
# ---------------------------- #
loss, accuracy = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")

model.save('better_emotion_model.h5')
print("✅ Model Saved Successfully!")