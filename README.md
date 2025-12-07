Skin Condition Detection for Low-Resolution ImagesThis project implements a Convolutional Neural Network (CNN) using PyTorch and 
Flask to classify seven common skin conditions from low-resolution images (28x28 pixels). 
It includes a training script, a web application for inference, and utilizes Grad-CAM for visual model interpretability. 

Dataset Information: 
1. HAM10000: This project is trained on the HAM10000 ("Human Against Machine" 10000) dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

Data Source: The raw images and associated metadata are available from the official academic sources (e.g., Harvard Dataverse).

Data Format Used: The train.py script specifically utilizes the aggregated CSV format of the 28x28 pixel RGB data, titled hmnist_28_28_RGB.csv.

Download Required: Due to its size and external source, the dataset file hmnist_28_28_RGB.csv is not included in this repository. To run the train.py script, you must manually download this file and place it in a folder named HAM10000/ in the project's root directory.

2. Pre-trained Model (skin_model.pth)
To simplify deployment and allow users to immediately run the web application (app.py), the trained model weights are included in the repository.

File: The file is named skin_model.pth.

Function: This file contains the state dictionary of the SkinCNN model trained using the settings in train.py (10 epochs on the HAM10000 dataset).

Ready for Inference: The app.py script directly loads this file: model.load_state_dict(torch.load("skin_model.pth", ...)). You do not need to run train.py to use the Flask web application, unless you wish to retrain the model or test changes.

Features:
1. 7-Class Classification: Detects and classifies seven different skin lesion categories from the HAM10000 dataset.
2. Low-Resolution Optimization: Specifically trained for 28x28 pixel images, common in medical datasets.
3. Web Interface (Flask): Easy-to-use web application for uploading images and getting real-time predictions.
4. Model Interpretability (Grad-CAM): Generates a heatmap overlaid on the image to show which regions the CNN focuses on for its decision.
5. Multilingual Support: Class names, danger levels, and medical advice are available in English (en), Spanish (es), and French (fr).
6. Medical Guidance: Provides an initial assessment of the danger level (Low, Medium, High) and general advice for each condition.

Project Structure:
The repository is organized as follows:skin-condition-detection/
├── HAM10000/
│   └── hmnist_28_28_RGB.csv  # The dataset file (not included, must be downloaded)
├── train.py                  # Script to train the CNN model
├── app.py                    # Flask application for the web interface
├── skin_model.pth            # Trained PyTorch model (generated after running train.py)
├── templates/
│   ├── index.html            # Main upload page
│   └── result.html           # Prediction results page
├── static/
│   └── ...                   # CSS, JS, and images
└── README.md                 # This file

Getting Started
1. PrerequisitesMake sure you have Python (3.x recommended) installed.
2. Setup:
Clone the repository and install the required Python packages:
Bash
git clone <repository-url>
cd skin-condition-detection
pip install -r requirements.txt
A decent accuracy will be printed upon completion (e.g., Test Accuracy: 80.00%).5. Running the Web AppStart the Flask server:Bashpython app.py
Open your web browser and navigate to http://127.0.0.1:5000/. You can now upload a skin lesion image (preferably cropped) and select a language to get a prediction, confidence scores, and the Grad-CAM visualization.🔍 
The Model (SkinCNN):
The core classification model is a simple Convolutional Neural Network (CNN) designed to work efficiently with the small 28x28 input size.

Architecture:
1. Conv1: 3 -> 32 filters, 3 X 3 kernel, ReLU activation.
2. MaxPool: 2 X 2.
3. Conv2: 32 -> 64 filters, 3 X 3 kernel, ReLU activation.
4. MaxPool: 2 X 2.
5. Flatten: Converts the output from 64 X 7 X 7 to a vector.
6. FC1: 64 X 7 X 7 -> 128 nodes, ReLU activation.
7. FC2: 128 -> 7$ nodes (Output classes).

The Grad-CAM implementation targets the conv2 layer to generate activation heatmaps, providing transparency on the model's decision-making process.
