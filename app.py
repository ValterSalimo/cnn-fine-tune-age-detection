import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
MODEL_PATH = "age_prediction_model.h5"
model = load_model(MODEL_PATH)

# Image size (same as used during training)
IMG_SIZE = (128, 128)

# Min and Max age for denormalization (update these with your dataset's min and max)
age_min = 0  # replace with your actual min age
age_max = 100  # replace with your actual max age

def preprocess_image(image_path):
    """Load and preprocess the image for model prediction."""
    img = load_img(image_path, target_size=IMG_SIZE)  # Load image
    img = img_to_array(img) / 255.0  # Convert to array & normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def denormalize_age(prediction, age_min, age_max):
    """Denormalize the predicted age back to the original scale."""
    return prediction * (age_max - age_min) + age_min

def predict_age(image_path):
    """Predict the age of the person in the image."""
    # Preprocess the image
    img = preprocess_image(image_path)

    # Predict age
    predicted_age = model.predict(img)[0][0]

    # Denormalize the prediction
    predicted_age = denormalize_age(predicted_age, age_min, age_max)

    return predicted_age

if __name__ == "__main__":
    # Test image path
    image_path = input("Enter the path of the image: ")  # Example: "/content/test_image.jpg"

    # Check if the file exists
    if os.path.exists(image_path):
        # Get prediction
        age = predict_age(image_path)
        print(f"The predicted age is: {age:.2f} years")
    else:
        print(f"Error: The image path '{image_path}' does not exist.")
