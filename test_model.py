# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix

# # === CONFIGURATION ===
# TEST_DIR = "test/Testing"  # or use full path if needed
# MODEL_PATH = "model/Ashu-Tumor-Model.h5"
# IMAGE_SIZE = (128, 128)
# BATCH_SIZE = 16

# # === LOAD DATA ===
# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#     TEST_DIR,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False
# )

# # === LOAD MODEL ===
# model = load_model(MODEL_PATH)
# print("Model loaded.")

# # === PREDICT ===
# preds = model.predict(test_generator, verbose=1)
# y_pred = np.argmax(preds, axis=1)
# y_true = test_generator.classes
# class_labels = list(test_generator.class_indices.keys())

# # === REPORT ===
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_true, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=class_labels))

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(BASE_DIR, "test", "Testing")
MODEL_PATH = os.path.join(BASE_DIR, "model", "Tumor-Model.h5")
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, "predictions.csv")
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")
GRADCAM_DIR = os.path.join(BASE_DIR, "gradcam_outputs")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

# === LOAD TEST DATA ===
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
print("Model loaded.")

# === PREDICT ===
preds = model.predict(test_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
filenames = test_generator.filenames

# === REPORT & CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_labels)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# === SAVE CONFUSION MATRIX PLOT ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(CONF_MATRIX_PATH)
plt.close()
print(f"Confusion matrix saved to {CONF_MATRIX_PATH}")

# === SAVE PREDICTIONS TO CSV ===
predicted_labels = [class_labels[i] for i in y_pred]
true_labels = [class_labels[i] for i in y_true]

df = pd.DataFrame({
    'Filename': filenames,
    'Predicted': predicted_labels,
    'True': true_labels
})
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"Predictions saved to {CSV_OUTPUT_PATH}")

# === GRAD-CAM ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# === SAVE GRAD-CAM IMAGES ===
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Find last conv layer automatically
last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer = layer.name
        break

print(f"Using last conv layer: {last_conv_layer}")

for i in range(min(10, len(filenames))):  # first 10 samples
    img_path = os.path.join(TEST_DIR, filenames[i])
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    # Superimpose heatmap
    raw_img = cv2.imread(img_path)
    raw_img = cv2.resize(raw_img, IMAGE_SIZE)
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(raw_img, 0.6, heatmap_color, 0.4, 0)

    pred_label = predicted_labels[i]
    true_label = true_labels[i]
    out_name = f"{i+1:02d}_{true_label}_pred_{pred_label}.jpg"
    out_path = os.path.join(GRADCAM_DIR, out_name)
    cv2.imwrite(out_path, overlay_img)

print(f"Saved Grad-CAM images to: {GRADCAM_DIR}/")


