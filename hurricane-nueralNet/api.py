from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model_name = "alt_lenet"
model_path = f'models/{model_name}.keras' # example, change for my model
model = tf.keras.models.load_model(model_path)

@app.route('/models/damage/v1', methods=['GET']) # change
def model_info():
   return {
       "version": "v1",
       "name": model_name,
       "description": f"Classify images as damage or not using {model_name} tensorflow model",
       "Total parameters:": model.count_params(),
       "Trainable parameters:": sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights]),
       "Non-trainable parameters:": sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
   }

###### make it able to accept images as jpegs. 

def preprocess_input(image):
    """
    Converts user-provided JPEG image into an array that can be used with the model.
    """
    try:
        image = image.resize((128, 128))
        image_array = np.array(image)
        image_array = image_array / 255.0
    except Exception as e:
        return {"error": f"{e}"}, 404
    
    # Add a batch dimension
    return image_array

@app.route('/models/damage/v1', methods=['POST'])
def classify_clothes_image():
    if 'image' not in request.files:
        return {"error": "The `image` field is required"}, 404
    file = request.files['image']
    if file:
        try:
            image = Image.open(file.stream)
            data = preprocess_input(image)
            prediction = model.predict(data)
            return {"result": prediction.tolist()}
        except Exception as e:
            return {"error": f"Could not process the image; {e}"}, 500
    return {"error": "No file provided"}, 400


@app.route('/models/damage/changeModel', methods=['POST'])
def change_model():
   global model
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


# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')