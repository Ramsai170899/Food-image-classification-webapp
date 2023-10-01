import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)


# Create a dictionary to map class labels to item names
class_label_to_item = {
    "Class1": "apples",
    "Class2": "baked potato",
    "Class3": "burger",
    "Class4": "coffee",
    "Class5": "crackers",
    "Class6": "curry",
    "Class7": "donut",
    "Class8": "durian",
    "Class9": "egg ballado",
    "Class10": "french fries",
    "Class11": "fried chicken",
    "Class12": "fried rice",
    "Class13": "gado gado",
    "Class14": "grapes",
    "Class15": "hot dog",
    "Class16": "ice cream",
    "Class17": "indomie",
    "Class18": "kebab",
    "Class19": "meatballs",
    "Class20": "nuts",
    "Class21": "omelette",
    "Class22": "oranges",
    "Class23": "pizza",
    "Class24": "porridge",
    "Class25": "rendang",
    "Class26": "sandwich",
    "Class27": "satay",
    "Class28": "soto",
    "Class29": "taco",
    "Class30": "water"
}


# Load the trained model
model = load_model('food_image_classifier_model.h5')

# Define a function to preprocess the image for prediction


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img


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

    if file:
        # Create the 'static/uploaded_images' directory if it doesn't exist
        os.makedirs('static/uploaded_images', exist_ok=True)

        # Save the uploaded file to the 'uploaded_images' folder
        filename = os.path.join('static', 'uploaded_images', file.filename)
        file.save(filename)

        # Preprocess the uploaded image
        processed_image = preprocess_image(filename)

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        class_probabilities = prediction.tolist()[0]

        # Create a list of class labels corresponding to your dataset
        class_labels = list(class_label_to_item.keys())

        # Map the predicted class label to the item name using the dictionary
        predicted_item = class_label_to_item.get(
            class_labels[predicted_class], "Unknown")

        response = {
            'class_label': predicted_item,
        }

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
