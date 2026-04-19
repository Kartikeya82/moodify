import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D, Reshape, Multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import classification_report

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

test_generator = val_datagen.flow_from_directory(
    os.path.join(output_dir, "test"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, "train"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(output_dir, "validation"),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
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
#  4️⃣ BUILD INDIVIDUAL MODELS
# ---------------------------- #
def build_mobilenet_se(img_shape=(96, 96, 3)):
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=img_shape)
    x = base.output
    x = se_block(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(7, activation='softmax')(x)
    model = Model(base.input, out)
    return model, base

def build_efficientnet(img_shape=(96, 96, 3)):
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=img_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(7, activation='softmax')(x)
    model = Model(base.input, out)
    return model, base

# ---------------------------- #
#  5️⃣ ENSEMBLE MODEL
# ---------------------------- #
mobilenet_model, mobilenet_base = build_mobilenet_se()
efficientnet_model, efficientnet_base = build_efficientnet()

mobilenet_model.compile(optimizer=AdamW(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer=AdamW(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

print("✅ Training MobileNetV2 with SE")
history_m = mobilenet_model.fit(train_generator, validation_data=val_generator, epochs=10)
print("✅ Training EfficientNetB0")
history_e = efficientnet_model.fit(train_generator, validation_data=val_generator, epochs=10)

# ---------------------------- #
#  6️⃣ FINE-TUNE BOTH MODELS
# ---------------------------- #
mobilenet_base.trainable = True
for layer in mobilenet_base.layers[:-100]:
    layer.trainable = False

mobilenet_model.compile(optimizer=AdamW(3e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("🔁 Fine-tuning MobileNetV2 with SE")
history_m_ft = mobilenet_model.fit(train_generator, validation_data=val_generator, epochs=10,
                    callbacks=[EarlyStopping(patience=4, restore_best_weights=True),
                               ReduceLROnPlateau(patience=2)])

efficientnet_base.trainable = True
for layer in efficientnet_base.layers[:-100]:
    layer.trainable = False

efficientnet_model.compile(optimizer=AdamW(3e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("🔁 Fine-tuning EfficientNetB0")
history_e_ft = efficientnet_model.fit(train_generator, validation_data=val_generator, epochs=10,
                       callbacks=[EarlyStopping(patience=4, restore_best_weights=True),
                                  ReduceLROnPlateau(patience=2)])

# ---------------------------- #
#  7️⃣ EVALUATE ENSEMBLE
# ---------------------------- #
print("✅ Evaluating Ensemble")
preds1 = mobilenet_model.predict(val_generator)
preds2 = efficientnet_model.predict(val_generator)
ensemble_preds = (preds1 + preds2) / 2

true_labels = val_generator.classes
pred_labels = np.argmax(ensemble_preds, axis=1)
acc = np.mean(pred_labels == true_labels[:len(pred_labels)])
print(f"\n✅ Ensemble Validation Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(true_labels[:len(pred_labels)], pred_labels, target_names=list(emotion_labels.values())))

# ---------------------------- #
#  8️⃣ TEST SET EVALUATION
# ---------------------------- #
print("\n🔍 Evaluating on Test Set")
test_preds1 = mobilenet_model.predict(test_generator)
test_preds2 = efficientnet_model.predict(test_generator)
test_ensemble_preds = (test_preds1 + test_preds2) / 2
test_true_labels = test_generator.classes
test_pred_labels = np.argmax(test_ensemble_preds, axis=1)
test_acc = np.mean(test_pred_labels == test_true_labels[:len(test_pred_labels)])
print(f"✅ Test Accuracy (Ensemble): {test_acc * 100:.2f}%")

# ---------------------------- #
#  9️⃣ SAVE MODELS
# ---------------------------- #
mobilenet_model.save("mobilenet_se_model.h5")
efficientnet_model.save("efficientnet_model.h5")
print("✅ Models saved!")

# ---------------------------- #
# 🔟 PLOT TRAINING CURVES
# ---------------------------- #
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.show()

plot_history(history_m, "MobileNetV2 Initial")
plot_history(history_e, "EfficientNetB0 Initial")
plot_history(history_m_ft, "MobileNetV2 Fine-tuned")
plot_history(history_e_ft, "EfficientNetB0 Fine-tuned")
