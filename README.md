Child_Find
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------

### First: About The Project
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
### How To Use Our Deployment:


**1- First you need to open this link**

![image](https://github.com/user-attachments/assets/18d10a1c-23ae-424c-9485-cf55d3d3f4c7)



**2- Second you need to enter the password**

![image](https://github.com/user-attachments/assets/d5bf4eb5-cc87-4d4a-8949-12c4e3a9fcdc)



**3-Third upload the image you need to know if it matches another image or not**

![image](https://github.com/user-attachments/assets/5ab98b83-6440-4cf2-b668-df36a272ca4d)



**4-Fourth if the image not match other images , added the image to the dataset**

![image](https://github.com/user-attachments/assets/07fd4c75-a012-4eaa-8f47-bcffd1e7354b)



**5-Fifth if the image matched another image , say that this image matched to image in our dataset**


![image](https://github.com/Aman-Organization-5leha-3la-Allah/AMAN_AI_Section/blob/main/Matched_images/matched_.jpg)








