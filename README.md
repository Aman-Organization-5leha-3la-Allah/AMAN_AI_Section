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


AI API Documentation

This documentation provides an overview of the AI API designed for a facial recognition system. The system allows users to upload images to build a training dataset and then upload a test image to compare with all images in the training dataset using a pre-trained FaceNet model.

Overview 

Purpose: This API allows users to add images to a training dataset and compare a test image with the training images to calculate similarity scores.

Libraries Used: Flask, Keras-FaceNet, TensorFlow, SciPy.

Endpoints:

/ - Home page with forms to upload training and test images.

/add_to_training - Endpoint to add images to the training dataset.

/compare - Endpoint to compare a test image with all training images.



Requirements

Ensure the following packages are installed:

pip install flask keras-facenet tensorflow scipy

File Structure

1. app.py: The main Flask application file, defining routes and implementing the functionality for adding images to the training dataset and comparing images.


2. index.html: The HTML file that provides a user interface for uploading images for training and testing.



API Endpoints

1. Home Page (/)

Method: GET

Description: Renders the home page with forms to upload training images and a test image.

Response: Renders index.html.


2. Add to Training (/add_to_training)

Method: POST

Description: Adds uploaded images to the training dataset, extracting embeddings using the FaceNet model and saving them for future comparisons.

Parameters:

train_images: Multiple image files to add to the training dataset.


Responses:

Success:

{
  "message": "Training images added successfully."
}

Error:

{
  "error": "No images uploaded"
}



3. Compare Test Image (/compare)

Method: POST

Description: Compares an uploaded test image with all images in the training dataset, calculating similarity scores based on Euclidean distance between embeddings.

Parameters:

test_image: A single image file to compare against the training dataset.


Responses:

Success:

{
  "similarities": [
    "Similarity with image1: 0.1234",
    "Similarity with image2: 0.5678",
    ...
  ]
}

Error:

{
  "error": "No test image uploaded"
}



Detailed Code Explanation

1. Flask Application (app.py)

Initialization:

Initializes the Flask app and loads the FaceNet model.

Sets up directories for storing training images and embeddings.


Helper Functions:

allowed_file(filename): Checks if the uploaded file has an allowed extension (png, jpg, jpeg).

preprocess_image(img_path): Preprocesses images to the correct size (160x160) and normalizes them for the FaceNet model.

get_embedding(img_path): Generates an embedding vector (128-dimensional) for the given image using the FaceNet model.


Routes:

/ Route: Renders index.html which contains forms for uploading training and test images.

/add_to_training Route:

Accepts multiple images for training.

Saves the images and extracts embeddings to add to the dataset.

Saves embeddings and image names to .npy files.


/compare Route:

Accepts a test image and extracts its embedding.

Compares the test embedding with all training embeddings using Euclidean distance.

Returns similarity scores indicating how closely the test image matches each training image.




2. HTML Interface (index.html)

Description: Provides a user-friendly interface for uploading images for training and testing.

Form Elements:

Training Form: Allows users to upload multiple images to add to the training dataset.

Testing Form: Allows users to upload a test image to compare against the training dataset.


JavaScript:

Handles form submissions asynchronously, sending requests to the API endpoints and displaying the results.
