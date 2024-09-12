###FACE FIND
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------

### About The Project
The thought of a family member, a friend or someone else you care about going missing can be terrifying. This project aims to help find your loved ones using Face Recognition Technology. If someone you know is missing, then,

Register the missing person with us.
Once the background check is done and the missing person is verified, we generate a unique Face ID for the missing person using Azure's Face API.
When volunteers report a suspected missing person, we verify and generate a Face ID the same way. We then use Azure's Find Similar API to identify a potential match with our database of missing person Face IDs.
If a match is found we will contact you.


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




