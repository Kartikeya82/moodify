from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Load the trained model
model = load_model('better_emotion_model.h5')
print("✅ Model Loaded Successfully!")

# ✅ Define test data directory
test_dir = r"D:\moodify\fer2013_images\test"

# ✅ Preprocess test images (convert to grayscale)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),  
    batch_size=16,
    class_mode='categorical',
    color_mode="rgb",  # ✅ Convert to grayscale
    shuffle=False
)

print("✅ Test Data Loaded!")

# ✅ Evaluate the model
loss, accuracy = model.evaluate(test_generator)

print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")


