from django.shortcuts import render, HttpResponse

# Create your views here.
import os
import numpy as np
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from django.core.files.storage import default_storage
from PIL import Image

# Load the model once when the server starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_recognition_CNN.h5")
model = load_model(MODEL_PATH)


@csrf_exempt
def predict_face(request):
    if request.method == "POST" and request.FILES.get("image"):
        # Save uploaded file temporarily
        file = request.FILES["image"]
        file_path = default_storage.save(file.name, file)
        img_path = default_storage.path(file_path)

        # Preprocess the image
        image = cv2.imread(img_path)
        if image is None:
            return JsonResponse({"error": "Invalid image"}, status=400)

        image = cv2.resize(image, (244, 244))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        prediction = model.predict(image)
        class_index = int(np.argmax(prediction, axis=1)[0])

        # Clean up
        default_storage.delete(file_path)

        return JsonResponse({"class": class_index})
    return JsonResponse({"error": "Invalid request"}, status=400)

    
# page html 
def index(request):
    return render(request, "index.html")
