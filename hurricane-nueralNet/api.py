from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)

model_name = "alt_lenet"
model_path = f'models/{model_name}.keras' # example, change for my model
model = tf.keras.models.load_model(model_path)

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
       "description": f"Classify images of houses as damaged or not using the {model_name} tensorflow model.",
       "total parameters:": str(model.count_params()),
       "trainable parameters:": str(sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])),
       "non-trainable parameters:": str(sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])),
   }

###### make it able to accept images as jpegs. 

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

@app.route('/model/predict', methods=['POST'])
def predict():
    """
    Classifies an image uploaded by the user as either "damaged" or "not damaged".
    
    POST: Expects a multipart/form-data request with an image file under the key "image".
    Returns a JSON object with the classification result and a qualitative assessment of confidence.
    
    Example:curl -X POST -F "image=@./data/damage/-93.795_30.03779.jpeg" http://localhost:5001/model/predict
    """
    global model
    if 'image' not in request.files:
        return {"error": "The `image` field is required"}, 404
    file = request.files['image']
    if file:
        try:
            image = Image.open(file.stream)
            data = preprocess_input(image)
            prediction = model.predict(data)
            prediction_in_words = "damaged" if (prediction > 0.5) else "not damaged"
            adjective = "unsure"
            if (prediction > 0.8) or (prediction < 0.2):
                adjective = "confident"
            elif (prediction > 0.6) or (prediction < 0.4):
                adjective = "somewhat confident"
            
            return {"result": prediction.tolist(), "Outcome": f"The model is {adjective} that the building that is {prediction_in_words}."}
        except Exception as e:
            return {"error": f"Could not process the image; {e}"}, 500
    return {"error": "No file provided"}, 400


@app.route('/changeModel', methods=['POST'])
def change_model():
    """
    Changes the TensorFlow model used by the server.

    POST: Expects a JSON object with a key "model_name" that specifies the name of the new model to load.
    Returns a JSON object with a message indicating successful model change and the path to the new model.

    Example: curl -X POST -H "Content-Type: application/json" -d '{"model_name": "vgg"}' http://127.0.0.1:5001/changeModel
    """
    global model, model_name, model_path
    model_name = request.json.get('model_name')
    if not model_name:
      return {
         "error": 
         "The `model_name` field is required\nYour options are: "}, 400 ### List available models here

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
    print(model.summary())
    return jsonify(model_summary)


# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')