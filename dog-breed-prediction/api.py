from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import pickle
import sklearn
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)

model_name = "inception"
model_path = f'models/{model_name}.keras' # example, change for my model
model = tf.keras.models.load_model(model_path)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/model/info', methods=['GET']) # change
def model_info():
    """
    Provides basic information about the currently loaded TensorFlow model.
    
    GET: Returns a JSON object with the model's version, name, description,
    total number of parameters, number of trainable parameters, and number of non-trainable parameters.

    Example: curl http://127.0.0.1:5001/model/info
    """
    global model
    return {
       "version": "v1",
       "name": str(model_name),
       "description": f"Predict the breed of a dog using the {model_name} tensorflow model.",
       "total parameters:": str(model.count_params()),
       "trainable parameters:": str(sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])),
       "non-trainable parameters:": str(sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])),
   }


@app.route('/model/models', methods=['GET'])
def list_available_models():
    """
    Lists the available TensorFlow models that users can switch to.
    
    GET: Returns a JSON object listing all available models and indicating the default model.

    Example: curl http://127.0.0.1:5001/model/models
    """
    available_models = ["vgg16", "inception", "lenet5"]
    return jsonify({
        "available_models": available_models,
        "default_model": "inception",
        "currently_loaded": model_name,
        "note": "inception is the default model loaded in and is the most accurate."
    })

def preprocess_input(image):
    """
    Converts user-provided JPEG image into an array that can be used with the model.
    """
    # check if image is right size
    try:
        image_array = np.array(image)
        image_array = image_array / 255.0
        image = image.convert('RGB')
        image_array = np.expand_dims(image_array, axis=0) # batch size (how many images) bc model expects (1, 128, 128, 3) not just (128, 128, 3)
    except Exception as e:
        return {"error": f"{e}"}, 404
    
    # Add a batch dimension
    return image_array

def predict():
    """
    Predicts the breed of a dog given an image uploaded by the user.
    
    POST: Expects a multipart/form-data request with an image file under the key "image".
    Returns a JSON object with the classification result.
    
    Example:curl -X POST -F "image=@./data/images/affenpinscher-1.jpg" http://localhost:5001/model/predict
    """
    if 'image' not in request.files:
        return jsonify({"error": "The `image` field is required"}), 404
    file = request.files['image']
    if file:
        try:
            # Open the image file and prepare it for the model
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((128, 128))  # Adjust size according to your model requirements
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Ensure this matches model's expected preprocessing

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction[0])
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            return jsonify({"result": predicted_label})
        except Exception as e:
            return jsonify({"error": f"Could not process the image; {str(e)}"}), 500
    return jsonify({"error": "No file provided"}), 400


@app.route('/model/change', methods=['POST'])
def change_model():
    """
    Changes the TensorFlow model used by the server.

    POST: Expects a JSON object with a key "model_name" that specifies the name of the new model to load.
    Returns a JSON object with a message indicating successful model change and the path to the new model.

    Example: curl -X POST -H "Content-Type: application/json" -d '{"model_name": "lenet5"}' http://127.0.0.1:5001/changeModel
    """
    global model, model_name, model_path
    available_models = ["vgg", "lenet5", "inception"]
    model_name = request.json.get('model_name')
    if not model_name:
      return { 
          "error":"The `model_name` field is required", 
          "available_models": available_models,
          "default_model": "inception",
          "currently_loaded": model_name,
          "note": "inception is the default model loaded in and is the most accurate."}, 400 

    if os.path.exists(f'models/{model_name}.keras'):
       new_model_path = f'models/{model_name}.keras'
    else:
       return {"error": "Model not found"}, 404
   
    try:
      model = tf.keras.models.load_model(new_model_path)
    except Exception as e:
      return {"error": f"Failed to load the model;\n {e}"}, 500
   
    return jsonify({
      "message": f"Model changed to {model_name} successfully",
      "model_path": new_model_path
    })



def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def model_summary_to_json(model):
    summary_string = get_model_summary(model)
    summary_lines = summary_string.split('\n')
    return summary_lines

@app.route('/model/summary', methods=['GET'])
def model_summary():
    """
    Provides a textual summary of the currently loaded TensorFlow model's architecture.
    
    GET: Returns a JSON array where each element is a string describing a layer in the model's architecture.

    Example: curl http://127.0.0.1:5001/model/summary
    """
    model_summary = model_summary_to_json(model)
    return jsonify(model_summary)


@app.route('/help', methods=['GET'])
def api_help():
    """
    Provides an overview and usage examples for the available API endpoints.
    """
    help_info = {
        "/model/info": {
            "method": "GET",
            "description": "Provides basic information about the currently loaded TensorFlow model.",
            "example": "curl http://127.0.0.1:5001/model/info"
        },
        "/model/predict": {
            "method": "POST",
            "description": "Predicts the breed of a dog given an image uploaded by the user.",
            "example": "curl -X POST -F 'image=@./data/images/affenpinscher-1.jpg' http://localhost:5001/model/predict"
        },
        "/model/change": {
            "method": "POST",
            "description": "Changes the TensorFlow model used by the server.",
            "example": "curl -X POST -H 'Content-Type: application/json' -d '{\"model_name\": \"vgg\"}' http://127.0.0.1:5001/model/change"
        },
        "/model/summary": {
            "method": "GET",
            "description": "Provides a textual summary of the currently loaded TensorFlow model's architecture.",
            "example": "curl http://127.0.0.1:5001/model/summary"
        },
        "/model/models": {
            "method": "GET",
            "description": "Lists the available TensorFlow models that users can switch to.",
            "example": "curl http://127.0.0.1:5001/model/models"
        }
    }
    return jsonify(help_info)

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')