# Create your views here.
# detection/views.py

from django.shortcuts import render
from django.apps import apps
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess_image_django(image_file, target_size=(299, 299)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(request):
    if request.method == "POST" and "image" in request.FILES:
        try:
            img_array = preprocess_image_django(request.FILES["image"])

            detection_config = apps.get_app_config('detection')
            model = detection_config.model

            if model:
                prediction = model.predict(img_array)[0][0]
                result = "Malignant" if prediction > 0.5 else "Benign"

                return render(request, "result.html", {"prediction": result})
            else:
                return render(request, "result.html", {"error": "Model is not loaded."})
        except Exception as e:
            return render(request, "result.html", {"error": f"Error processing image: {str(e)}"})

    return render(request, "result.html")


def home(request):
    return render(request, 'home.html') 