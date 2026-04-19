import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D, Reshape, Multiply)
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

# Create image folders
usage_map = {"Training": "train", "PublicTest": "validation", "PrivateTest": "test"}
df = pd.read_csv(csv_file)

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
#  2️⃣ LOAD DATASET
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
    os.path.join(output_dir, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(output_dir, "validation"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# ---------------------------- #
#  3️⃣ Squeeze-and-Excitation (SE) Block
# ---------------------------- #
def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal')(se)
    return Multiply()([input_tensor, se])

# ---------------------------- #
#  4️⃣ BUILD MODEL WITH SE ATTENTION
# ---------------------------- #
def build_attention_model(img_shape=(96, 96, 3)):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=img_shape)
    base_model.trainable = False

    x = base_model.output
    x = se_block(x)  # ✅ Add SE attention block
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(7, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=output)

model = build_attention_model()
model.compile(optimizer=AdamW(learning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy"])

print("✅ Model with SE attention built successfully!")

# ---------------------------- #
#  5️⃣ TRAIN PHASE 1
# ---------------------------- #
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# ---------------------------- #
#  6️⃣ FINE-TUNE LAST LAYERS
# ---------------------------- #
model.trainable = True
for layer in model.layers[:-100]:
    layer.trainable = False

model.compile(optimizer=AdamW(learning_rate=3e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=["accuracy"])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=[early_stopping, reduce_lr]
)

# ---------------------------- #
#  7️⃣ FINAL EVALUATION
# ---------------------------- #
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
print(f"✅ Final Validation Loss: {val_loss:.4f}")

# ---------------------------- #
#  8️⃣ SAVE MODEL
# ---------------------------- #
model.save("emotion_model_se_attention.h5")
print("✅ Model Saved Successfully!")
