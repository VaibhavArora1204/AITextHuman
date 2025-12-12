from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # required to load some TF-Hub BERT models
import os

app = Flask(__name__)

# -------- Model loading --------
# Option 1: load a SavedModel you exported from the notebook:
MODEL_PATH = os.path.join("models", "bert_model")

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "KerasLayer": hub.KerasLayer
        }
    )
    print("Loaded saved BERT model from", MODEL_PATH)
except Exception as e:
    print("Could not load saved model, falling back to building from TF Hub:", e)
    # Option 2: build a fresh model consistent with your notebook setup
    # NOTE: adjust the handles to match what you used in the .ipynb
    preprocess_handle = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_handle = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(preprocess_handle, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(encoder_handle, trainable=False, name="BERT_encoder")
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]

    x = tf.keras.layers.Dropout(0.3)(pooled_output)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    logits = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = tf.keras.Model(text_input, logits)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    # At this point you should load your trained weights if you saved them:
    # model.load_weights("models/bert_weights.h5")

# -------- Routes --------

@app.route("/")
def index():
    # Renders templates/index.html
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON body: { "text": "some essay ..." }
    Response: { "probability": 0.87, "label": 1 }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Model expects a batch; wrap in list
        pred = model.predict([text], verbose=0)
        prob = float(pred[0][0])
        label = int(prob >= 0.5)
        return jsonify({
            "probability": prob,
            "label": label
        })
    except Exception as e:
        # Log the error in real usage
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
