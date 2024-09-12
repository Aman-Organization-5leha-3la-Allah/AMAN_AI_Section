### First: Ai code Documentation

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

### Second: AI API Documentation
AI_API Documentation for Missing Children Recognition
Overview
This API allows users to upload images to find potential matches in a dataset of missing children. The AI model compares the uploaded image with the dataset and returns whether there is a match or if the image is new. If no match is found, the API can save the image into the dataset for future reference.

Base URL
The API is hosted locally and can be accessed via a public tunnel when running. For example:

vbnet
Copy code
https://<your-public-tunnel>.loca.lt
Authentication
Currently, the API does not require authentication for use, as it is designed for local or private use. If security is required in the future, consider implementing OAuth or an API key system.

Endpoints
1. Image Matching
POST /image/match
Upload an image to match against the missing children dataset.

Request:

Endpoint: /image/match
Method: POST
Headers:
json
Copy code
{
  "Content-Type": "multipart/form-data"
}
Body: You should upload an image file using a multipart/form-data request.
Response:

Success (200):

json
Copy code
{
  "match_found": true,
  "matched_name": "John Doe",
  "confidence_score": 0.95
}
If a match is found, the response will return the name of the missing child and a confidence score.

No Match (200):

json
Copy code
{
  "match_found": false,
  "message": "No match found. Image added to the dataset."
}
If no match is found, the image will be saved in the dataset for future queries.

2. Dataset Management (Future Feature)
This endpoint can be used for managing the dataset of missing children. Currently not implemented but possible enhancements could include:

GET /dataset/images
Retrieve all images stored in the dataset for review or management.

DELETE /dataset/image/{image_id}
Remove an image from the dataset.

Example Code
Python Example
python
Copy code
import requests

url = 'https://<your-public-tunnel>.loca.lt/image/match'
image_path = 'path/to/your/image.jpg'
files = {'file': open(image_path, 'rb')}

response = requests.post(url, files=files)

print(response.json())
Running Locally
To run the API locally and expose it using a public tunnel:

Install the required libraries (e.g., streamlit, localtunnel).
Run the following command:
bash
Copy code
streamlit run app.py & npx localtunnel --port 8501
Access the publicly available URL provided by localtunnel.

