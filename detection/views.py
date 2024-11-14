# Create your views here.

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
            image_file = request.FILES["image"]
            img_array = preprocess_image_django(image_file)

            detection_config = apps.get_app_config('detection')
            model = detection_config.model

            if model:
                prediction = model.predict(img_array)[0][0]
                result = "Malignant" if prediction > 0.5 else "Benign"
                image_file.seek(0)

                return render(request, "result.html", {"prediction": result, "image": image_file})
            else:
                return render(request, "result.html", {"error": "Model is not loaded."})
        except Exception as e:
            return render(request, "result.html", {"error": f"Error processing image: {str(e)}"})

    return render(request, "result.html")


def home(request):
    return render(request, 'home.html') 