# Agri Vision

Agri Vision is a simple crop health classification application that uses a pre-trained deep learning model to identify possible diseases from crop leaf images.

The application is designed to give farmers a quick preliminary assessment of crop health and provide simple guidance based on the detected condition.

All the work was done by me and my teammate, Ajohn, we just did it on the school computer so the PR was required.

https://github.com/user-attachments/assets/d34747e4-0481-42de-b469-03e264ff1c39

## Features

- Upload a crop leaf image
- Detect possible crop diseases using a trained deep learning model
- Supports pepper, potato, and tomato crops
- Displays the detected crop
- Displays the possible disease or healthy condition
- Shows prediction confidence
- Provides simple guidance based on the prediction
- Checks basic image quality before analysis
- Simple and accessible Streamlit interface

## Supported Crops

Agri Vision currently supports:

- Pepper Bell
- Potato
- Tomato

The model can identify 15 different crop conditions across these crops.

## How It Works

The application follows a simple process:

1. The user uploads a photo of a crop leaf.
2. The application checks the basic quality of the image.
3. The image is resized to the required input size.
4. The image is processed and passed to the trained deep learning model.
5. The model predicts the most likely crop condition.
6. The application displays the crop and possible disease.
7. The user receives simple guidance based on the prediction.

## Technology Used

- Python
- TensorFlow
- OpenCV
- NumPy
- Streamlit

## Model

Agri Vision uses a pre-trained TensorFlow model stored in the following file:

    crop_disease_model.h5

The model expects input images with a size of:

    224 x 224 pixels

The model predicts one of 15 possible crop conditions.

These include:

- Pepper Bell Bacterial Spot
- Pepper Bell Healthy
- Potato Early Blight
- Potato Late Blight
- Potato Healthy
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Healthy

## Image Guidelines

For the best results, users should:

- Take a clear photo of the crop leaf
- Use good natural lighting
- Make sure the leaf is clearly visible
- Avoid blurry images
- Show the affected part of the leaf when possible
- Avoid covering the leaf with their hand

The application performs basic image quality checks and may ask the user to take another photo if the image is too dark, too bright, or too blurry.

## Intended Users

Agri Vision is designed with accessibility and simplicity in mind.

The goal is to make preliminary crop disease information easier to access for farmers, including users who may have limited technical experience.

The application focuses on providing:

- Simple image-based interaction
- Clear crop and disease information
- Easy-to-understand guidance
- A straightforward user experience

## Disclaimer

Agri Vision provides an AI-based preliminary assessment of crop health.

The results should not be considered a definitive diagnosis or a replacement for professional agricultural advice.

Users should consult qualified agricultural experts before making important decisions about crop treatment or management.

## License

This project is intended for educational and demonstration purposes.
