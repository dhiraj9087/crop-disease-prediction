from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import uuid
import logging

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models with error handling
def load_model_safe(model_path, model_name):
    if not os.path.exists(model_path):
        logger.error(f"{model_name} file not found at: {model_path}")
        raise FileNotFoundError(f"{model_name} file not found at: {model_path}")
    try:
        model = load_model(model_path)
        logger.info(f"{model_name} loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {str(e)}")
        raise

try:
    unet_model = load_model_safe('models/unet_model.h5', 'UNet model')
    resnet_model = load_model_safe('models/resnet50_model.h5', 'ResNet50 model')
except Exception as e:
    logger.error(f"Application startup failed: {str(e)}")
    raise

# Define disease classes (38 classes from ResNet50 training)
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img.astype('float32') / 255.0
    return img, img_array

def generate_mask(image_array):
    img_array = cv2.resize(image_array, (128, 128))
    img_array = np.expand_dims(img_array, axis=0)
    mask = unet_model.predict(img_array)  # Shape: (1, 128, 128, 39)
    mask = np.argmax(mask, axis=-1)  # Shape: (1, 128, 128)
    mask = mask[0]  # Shape: (128, 128)
    mask = (mask > 0).astype(np.uint8) * 255  # Binary mask (0 or 255)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    return mask

def predict_disease(original_img, mask):
    mask_channel = mask.astype('float32') / 255.0
    mask_channel = mask_channel[..., np.newaxis]  # Shape: (224, 224, 1)
    img_array = original_img.astype('float32') / 255.0  # Shape: (224, 224, 3)
    combined_input = np.concatenate([img_array, mask_channel], axis=-1)  # Shape: (224, 224, 4)
    combined_input = np.expand_dims(combined_input, axis=0)  # Shape: (1, 224, 224, 4)
    prediction = resnet_model.predict(combined_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    return disease_classes[predicted_class], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"Uploaded image saved to: {file_path}")

        # Process image for UNet (128, 128) and ResNet50 (224, 224)
        _, img_array_unet = preprocess_image(file_path, target_size=(128, 128))
        original_img, _ = preprocess_image(file_path, target_size=(224, 224))
        mask = generate_mask(img_array_unet)
        
        # Save mask for display
        mask_filename = str(uuid.uuid4()) + '.png'
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
        cv2.imwrite(mask_path, mask)
        
        # Predict disease
        disease, confidence = predict_disease(original_img, mask)
        
        # Prepare response
        result = {
            'disease': disease,
            'confidence': f'{confidence:.2%}',
            'image_path': f'static/uploads/{filename}'
            # 'mask_path': mask_path.replace('static/', '')
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)