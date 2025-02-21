Age Prediction Model
====================

This project is a deep learning-based age prediction model using a pre-trained ResNet50V2 architecture. The model is trained on the UTKFace dataset and predicts the age of a person from an input image.

## Project Structure
- `age_prediction_model.h5` - Trained model file
- `app.py` - Script to load the trained model and make predictions
- `utkface_aligned_cropped/UTKFace/` - Directory containing the dataset

## Requirements
To run this project, install the following dependencies:
```
pip install tensorflow numpy
```

## Training
The model is trained using the UTKFace dataset. The dataset consists of face images labeled with age. Images are preprocessed, normalized, and augmented before being fed into a ResNet50V2-based model.

## How to Use
### 1. Load the Model and Predict Age
Run the `app.py` script:
```
python app.py
```
Enter the path of the image when prompted. The model will predict the age of the person in the image.

### 2. Model Input and Output
- **Input:** A face image with dimensions (128, 128, 3)
- **Output:** Predicted age (denormalized to original scale)

## Notes
- Ensure that the image path is correct.
- The model predictions may vary slightly due to data augmentation and normalization.
- The dataset should be preprocessed to match the training format.

## Future Improvements
- Fine-tuning the model on a larger dataset for better accuracy.
- Implementing a Flask-based API for age prediction.
- Optimizing the model for faster inference.

trained model:https://drive.google.com/drive/folders/12jI90bI1TXWOD_hRM2WQbLXAup4A4peF?usp=drive_link

## Author
Valter

