#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keras-facenet


# In[1]:


from keras_facenet import FaceNet


# In[9]:


from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from scipy.spatial.distance import euclidean

# Initialize Flask app
app = Flask(__name__)

# Assuming you have loaded your pre-trained FaceNet model
model = FaceNet()  # Load your FaceNet model

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image for prediction (for FaceNet)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))  # FaceNet typically uses 160x160 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0,1] scale
    return img_array

# Function to get FaceNet embedding from an image
def get_embedding(img_path):
    img = preprocess_image(img_path)
    embedding = model.predict(img)
    return embedding[0]  # Return the embedding (usually 128-dimensional vector)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and similarity prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Two files must be uploaded'})

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Both files must be selected'})

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        # Save the uploaded files to the "uploads" folder
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path1 = os.path.join(upload_folder, file1.filename)
        file_path2 = os.path.join(upload_folder, file2.filename)
        file1.save(file_path1)
        file2.save(file_path2)

        # Get embeddings for both images
        embedding1 = get_embedding(file_path1)
        embedding2 = get_embedding(file_path2)

        # Calculate the Euclidean distance between the two embeddings
        distance = euclidean(embedding1, embedding2)

        # Return the similarity score (lower distance means more similar)
        return jsonify({'similarity_score': float(distance)})

    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use a different port like 5001


# In[ ]:




