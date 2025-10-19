CIFAR-10 Image Classifier with ResNet18 and Web Interface
This project implements a complete pipeline for training and deploying a deep learning model for image classification. It uses a modified ResNet18 architecture to classify images from the 10 categories of the CIFAR-10 dataset.

The project is structured with separate components for:

Model Training: A script to train the model from scratch on the CIFAR-10 dataset.

API Backend: A Flask server that loads the trained model and exposes a REST API for predictions.

Web Frontend: A Streamlit web application that provides a user-friendly interface to upload images and view classification results.

(Note: This is a placeholder for a screenshot of your application)

Table of Contents
Project Structure

Features

Setup and Installation

How to Run

API Usage

Model Architecture

How to Contribute

Project Structure
.
├── flask_app.py        # Flask backend to serve the model via API
├── streamlit.py        # Streamlit frontend for user interaction
├── model.py            # PyTorch model definition (CIFARResNet18)
├── train.py            # Script to train the model
├── utils.py            # Helper functions, transforms, and class labels
├── predict_client.py   # Command-line client for testing the API
├── saved_models/       # Directory where trained models are saved
└── data/               # Directory where CIFAR-10 dataset will be downloaded
Features
End-to-End MLOps Pipeline: From training to deployment and user interaction.

Modified ResNet18: A torchvision.models.resnet18 architecture optimized for small 32x32 images.

RESTful API: A robust Flask backend serves model predictions, making it accessible to any client.

Interactive Web UI: A clean and simple Streamlit interface for easy image uploads and prediction visualization.

Modular Code: Well-structured and separated components for maintainability and scalability.

Setup and Installation
Follow these steps to set up the project environment.

1. Clone the Repository

Bash

git clone <your-repository-url>
cd <your-repository-directory>
2. Create a Python Virtual Environment (Recommended)

Bash

# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies Create a requirements.txt file with the following content:

Plaintext

torch
torchvision
flask
streamlit
requests
pillow
tqdm
Then, install the packages:

Bash

pip install -r requirements.txt
How to Run
Follow these steps in order to train the model and launch the applications.

Step 1: Train the Model
This script will download the CIFAR-10 dataset, train the CIFARResNet18 model, and save the best-performing version to saved_models/best_model.pth.

Bash

python train.py
You can adjust hyperparameters like epochs, lr, and batch_size directly in the train.py script.

Step 2: Prepare the Model for Serving
The Flask application looks for the model file. You can either move the trained model to the expected location or set an environment variable.

Option A: Move the Model (Recommended) The Flask app expects the model at models/resent.pth by default. Note the typo "resent.pth" in the default path in flask_app.py.

Bash

mkdir -p models
mv saved_models/best_model.pth models/resent.pth
Option B: Use an Environment Variable Set the Model_Path environment variable to point directly to your saved model file.

Bash

# For Linux/macOS
export Model_Path="saved_models/best_model.pth"

# For Windows (Command Prompt)
set Model_Path="saved_models\best_model.pth"
Step 3: Start the Flask API Backend
Run the Flask application to start the API server. The server will run on http://127.0.0.1:5000.

Bash

python flask_app.py
Your terminal should indicate that the server is running and ready to accept requests.

Step 4: Start the Streamlit Frontend
Open a new terminal (while the Flask server is still running) and run the Streamlit application.

Bash

streamlit run streamlit.py
This will automatically open a new tab in your web browser. You can now interact with the application.

Step 5: Make a Prediction
Navigate to the Streamlit URL provided in your terminal.

Click "Browse files" to upload an image.

Click the "Predict" button.

The application will display the uploaded image, the predicted class, and the probability scores for all 10 classes.

API Usage
The Flask backend provides two main endpoints.

Health Check
URL: /health

Method: GET

Description: A simple endpoint to verify that the API server is running.

Success Response (200 OK):

JSON

{
  "Status": "ok"
}
Prediction
URL: /predict

Method: POST

Description: Accepts an image file and returns the classification result.

Request Body: multipart/form-data with a file key containing the image.

Success Response (200 OK):

JSON

{
    "predicted_class": "dog",
    "predicted_index": 5,
    "probabilities": {
        "airplane": 0.0012,
        "automobile": 0.0003,
        "bird": 0.0151,
        "cat": 0.1420,
        "deer": 0.0438,
        "dog": 0.7511,
        "frog": 0.0215,
        "horse": 0.0119,
        "ship": 0.0041,
        "truck": 0.0090
    }
}
Error Responses (400 Bad Request):

If no file is uploaded: {"error": "no file uploaded"}

If the file is not a valid image: {"error": "invalid image", "detail": "<error_details>"}

Command-Line Client
You can also test the API using the predict_client.py script. Note: The provided predict_client.py has a hardcoded file path as the URL. You should modify it to point to the correct API endpoint.

Python

# In predict_client.py, change this line:
def predict(image_path, url='http://127.0.0.1:5000/predict'):
    # ... rest of the code
After correcting the URL, you can run it from the command line:

Bash

python predict_client.py /path/to/your/image.jpg
Model Architecture
The model is a ResNet18 from torchvision, specifically adapted for the 32x32 pixel images of the CIFAR-10 dataset. The standard ResNet18 is designed for larger images like those in ImageNet (224x224).

The key modifications in model.py are:

Modified First Convolutional Layer: The initial nn.Conv2d layer's kernel_size is changed from 7 to 3 and stride from 2 to 1. This prevents the model from losing too much spatial information from the small input images at the very first layer.

Initial Max-Pooling Removed: The self.model.maxpool layer is replaced with an nn.Identity() layer. This effectively removes the initial max-pooling step, which would otherwise reduce the feature map dimensions too aggressively for a 32x32 image.

Adapted Final Layer: The final fully connected layer (fc) is replaced with a new nn.Linear layer that outputs 10 logits, corresponding to the 10 classes in the CIFAR-10 dataset.
