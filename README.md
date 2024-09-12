### Ai code Documentation

#### 1. **Installation and Imports**
   - **Purpose:** Installs necessary libraries (`face_recognition` and `deepface`) and imports required modules.
   - **Code:**
     ```python
     !pip install face_recognition
     !pip install deepface

     import os
     import glob
     from PIL import Image, ImageDraw
     import matplotlib.pyplot as plt
     from deepface import DeepFace
     import cv2
     import dlib
     import numpy as np
     ```
   
### 2. **Model Section**

The facial recognition system uses DeepFace with two pre-trained models: VGG-Face and Facenet.

1-VGG-Face is based on the VGG-16 architecture, producing a 4096-dimensional embedding that captures facial features accurately.

2-Facenet employs a modified Inception network, generating a 128-dimensional vector optimized for face verification and clustering.


Both models are used to generate embeddings for input images, which are then compared against known faces using cosine similarity. By combining results from multiple models, the system improves accuracy and robustness, making it suitable for applications .



#### 3. **Helper Functions**
   - **Cosine Similarity Calculation:**
     - **Function:** `cosine_similarity(vec1, vec2)`
     - **Purpose:** Computes the cosine similarity between two vectors, used to measure similarity between face embeddings.
   
   - **Face Alignment:**
     - **Function:** `align_face(image_path)`
     - **Purpose:** Aligns the face in the input image using dlib's landmarks detector, which helps improve recognition accuracy.
   
   - **Image Preprocessing:**
     - **Function:** `preprocess_image(image_path)`
     - **Purpose:** Aligns, resizes, converts to grayscale, and normalizes images to prepare them for embedding generation.
   
   - **Display Images:**
     - **Function:** `display_images(test_image_path, matched_image_path, name, similarity)`
     - **Purpose:** Displays test and matched images side-by-side using Matplotlib for visual comparison.

#### 4. **Main Functionalities**
   - **Facial Search:**
     - **Function:** `facial_search(image_to_test, known_faces, known_face_names, models=['VGG-Face', 'Facenet'])`
     - **Purpose:** Compares a test image against known faces using embeddings from the specified models. Displays the matched image if a match is found above a set similarity threshold.

   - **Adding New Faces:**
     - **Function:** `add_to_training(image_path, known_faces, known_face_names)`
     - **Purpose:** Adds a new face image to the known faces training set, generates embeddings, and updates the saved data files.

#### 5. **Main Script Execution**
   - **Purpose:** Handles the main logic, including loading existing known faces, allowing users to upload new images for training, and processing test images for facial recognition.
   - **Key Steps:**
     1. Loads saved embeddings and face names.
     2. Allows users to upload images for training.
     3. Adds the uploaded images to the training set.
     4. Allows users to upload test images and runs the facial search function to find matches.

### Conclusion
our code provides an end-to-end implementation of a facial recognition system using `DeepFace` and `dlib`. It includes functionalities for face alignment, image preprocessing, similarity calculation, training data updates, and facial search with visual feedback on matched images.

-----------------------------------

### AI API Documentation

This documentation outlines the API for a facial recognition system built using Flask and FaceNet. The API supports uploading images to add to a training dataset and comparing a test image with all images in the dataset.


# Overview

Purpose:
To add images to a training dataset and compare a test image against the dataset using FaceNet embeddings.


# Technologies Used:

1. Flask: Web framework for building the API.

2. FaceNet: Pre-trained model for generating face embeddings.

3. TensorFlow and Keras: Libraries for model processing.

4. SciPy: For calculating Euclidean distance.


---

# Endpoints

1. Home Page (/)

Method: GET

Description: Renders the main page with forms for uploading images for training and testing.

Response: HTML page with forms.


2. Add to Training Dataset (/add_to_training)

 Method: POST

Description: Uploads images to be added to the training dataset.

Parameters:

train_images: Multiple image files to be added to the training set.


# Response:

Success:

`{
  "message": "Training images added successfully."
}`

Error:

`{
  "error": "Error message"
}`


Notes:

- Images are saved to the training_data directory.

- Embeddings and image names are stored in embeddings.npy and names.npy respectively.



3. Compare Test Image (/compare)

Method: POST

Description: Uploads a test image and compares it with all images in the training dataset.

Parameters:

test_image: The image file to be tested.


Response:

Success:

`{
  "similarities": [
    "Similarity with image1: 0.1234",
    "Similarity with image2: 0.5678"
  ]
}`

Error:

`{
  "error": "Error message"
}`


Notes:

Compares the test image with all stored embeddings and calculates Euclidean distances.

Returns a list of similarity scores for each image in the training dataset.


Detailed API Usage

Home Page (/)

Usage:

Access the URL where the Flask app is running (e.g., http://localhost:5001) to interact with the forms for training and testing images.



Add to Training Dataset (/add_to_training)

Request Example:

Form Data: Upload multiple image files under the train_images field.

Example cURL Request:

curl -X POST -F "train_images=@image1.jpg" -F "train_images=@image2.jpg" http://localhost:5001/add_to_training



Compare Test Image (/compare)

Request Example:

Form Data: Upload a single image file under the test_image field.

Example cURL Request:

curl -X POST -F "test_image=@test_image.jpg" http://localhost:5001/compare

Error Handling

File Not Found:

```json{
  "error": "File not found."
}```

Invalid File Type:

```json{
  "error": "Invalid file type. Only png, jpg, and jpeg are allowed."
}```

Missing Parameters:

``` json{
  "error": "Required parameters are missing."
}```



Summary

The AI API provides a way to dynamically add images to a facial recognition training dataset and compare new images against this dataset. It uses FaceNet for generating embeddings and calculates similarity based on Euclidean distance. The API is accessible through a web interface built with Flask, and results are returned in JSON format for ease of integration and analysis.


