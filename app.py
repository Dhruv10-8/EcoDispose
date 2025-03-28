import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ GitHub Personal Access Token for AI model
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN
)

app = Flask(__name__)
CORS(app)

# ✅ Path to TFLite model
model_path = "waste_classification_model.tflite"

# ✅ Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# ✅ Get input and output details for prediction
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Define class labels
categories = ["O", "R"]
category_map = {
    "O": "Organic",
    "R": "Recyclable"
}

@app.route("/classify", methods=["POST"])
def classify_waste():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # ✅ Load and preprocess image
        image = request.files["image"]
        image = Image.open(image).convert("RGB")
        image = image.resize((224, 224))  # Resize for the model
        image = np.array(image) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension

        # ✅ Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image)

        # ✅ Run inference
        interpreter.invoke()

        # ✅ Get prediction results
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = categories[predicted_class]

        # ✅ Map to Recyclable/Organic
        classification_type = category_map[predicted_label]

        # ✅ AI model for disposal suggestions
        prompt = f"The item is classified as {classification_type}. Provide sustainable and eco-friendly disposal suggestions. Return the response in plain text without any formatting, bold, italic, large fonts, etc."

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o",
            temperature=1,
            max_tokens=4096,
            top_p=1
        )
        suggestions = response.choices[0].message.content.strip()

        return jsonify({
            "prediction": classification_type,
            "suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
