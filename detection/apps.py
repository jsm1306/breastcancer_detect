import os
from django.apps import AppConfig
from tensorflow.keras.models import load_model

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'breastcancer_inception.keras')

    def ready(self):
        if not hasattr(self, 'model'):
            try:
                self.model = load_model(self.MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {e}")
