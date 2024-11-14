from django.shortcuts import render
from django.apps import apps
from PIL import Image as PilImage
import numpy as np
import base64
from io import BytesIO
from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess_image_django(image_file, target_size=(299, 299)):
    img = PilImage.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(request):
    if request.method == "POST" and "image" in request.FILES:
        try:
            img = PilImage.open(request.FILES["image"]).convert("RGB")
            
            img_array = preprocess_image_django(request.FILES["image"])

            detection_config = apps.get_app_config('detection')
            model = detection_config.model

            if model:
                prediction = model.predict(img_array)[0][0]
                result = "Malignant" if prediction > 0.5 else "Benign"
                
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                return render(request, "result.html", {
                    "prediction": result,
                    "image_data": img_base64
                })
            else:
                return render(request, "result.html", {"error": "Model is not loaded."})
        except Exception as e:
            return render(request, "result.html", {"error": f"Error processing image: {str(e)}"})

    return render(request, "result.html")
def home(request):
    return render(request, 'home.html') 