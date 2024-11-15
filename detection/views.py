import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tensorflow as tf
from django.shortcuts import render
from django.apps import apps
from tensorflow.keras.applications.inception_v3 import preprocess_input

def preprocess_image(image_file, target_size=(299, 299)):  # Change size to (299, 299)
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)  # Resize to (299, 299)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def clear_memory():
    tf.keras.backend.clear_session()

def predict_image(request):
    if request.method == "POST" and "image" in request.FILES:
        try:
            img = request.FILES["image"]
            img_array = preprocess_image(img)  # Now processes with (299, 299)

            detection_config = apps.get_app_config('detection')
            model = detection_config.model

            if model:
                prediction = model.predict(img_array)[0][0]
                result = "Malignant" if prediction > 0.5 else "Benign"

                buffered = BytesIO()
                img = Image.open(img)
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                clear_memory()

                return render(request, "result.html", {
                    "prediction": result,
                    "image_data": img_base64
                })
            else:
                return render(request, "result.html", {"error": "Model not loaded"})

        except Exception as e:
            return render(request, "result.html", {"error": f"Error processing image: {str(e)}"})

    return render(request, "result.html")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

def home(request):
    return render(request, 'home.html')
