# Breast Cancer Detection Using Deep Learning

This project uses a deep learning model to detect breast cancer from histopathological images. The model is based on a pre-trained Inception architecture and achieves an accuracy of 86%. 

## Project Overview

Breast cancer is one of the most common cancers affecting individuals worldwide. Early and accurate detection is critical for effective treatment and improving survival rates. This project aims to leverage the power of deep learning to detect cancerous cells from images, making it a valuable tool for aiding in early diagnosis.

### Model

The model used in this project is based on the **Inception** architecture, which has been fine-tuned to achieve an accuracy of 86%. Inception models are known for their ability to capture complex patterns in images and have been effective for various image classification tasks. 

### Dataset

This project was trained on the **BreakHis** dataset, a collection of breast cancer histopathology images that contain labeled samples of benign and malignant cells. The images were divided into training and testing sets, and data augmentation was used to enhance model performance.

## Project Structure

- **detection/**: The main Django app directory.
  - **models/**: Contains the trained model (excluded from GitHub due to file size restrictions).
  - **migrations/**: Django migrations for database schema.
  - **templates/**: HTML templates for the front-end, including pages for uploading images, viewing results, and more.
  - `views.py`: Defines the logic for handling image uploads, processing, and displaying results.
  - `urls.py`: Routes for different endpoints in the Django application.
  - `forms.py`: Contains forms for handling user input, such as uploading images.

- **manage.py**: Django’s management script for running the server, migrations, etc.
- **.gitignore**: Ignores the `models` folder to avoid pushing large model files to GitHub.

## Installation

To get started with this project, follow these steps:

### Prerequisites

- Python 3.8+
- Django 3.x+
- TensorFlow or Keras for loading and using the model
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jsm1306/breastcancer_detect.git
   cd breastcancer_detect
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Migrate the Database**:
   ```bash
   python manage.py migrate
   ```

4. **Run the Server**:
   ```bash
   python manage.py runserver
   ```

   The project should now be running on `http://127.0.0.1:8000`.

### Loading the Model

Since the model file is large, it is not included in the GitHub repository. Ensure that the trained model file (`breastcancer_inception.keras`) is placed in `detection/models/`. This model will be loaded by the application for making predictions.

## Usage

1. Access the application via the browser at `http://127.0.0.1:8000`.
2. Upload a histopathology image on the upload page.
3. Submit the image to receive a prediction on whether it shows signs of breast cancer.
4. The model will process the image and display the results on the results page.

## Model Performance

- **Architecture**: Inception
- **Accuracy**: 86%
- **Dataset**: BreakHis dataset

## Future Improvements

- **Improve Model Accuracy**: Experiment with more complex architectures or ensemble methods to increase the accuracy.
- **Deployment**: Deploy the model on a cloud service for better accessibility.
- **User Interface**: Improve the user interface for better usability and accessibility.
