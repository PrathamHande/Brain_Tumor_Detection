
# Brain Tumor Detection using Deep Learning

## Project Overview
This project focuses on the automatic detection and classification of brain tumors using deep learning techniques. MRI images are used as the primary data source, and multiple convolutional models have been trained and evaluated to identify and classify tumor types. The goal is to assist radiologists by providing an accurate and reliable computer-aided diagnosis system.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV
- Google Colab (for model training)
- Streamlit (for deployment interface)

## Dataset
We used a publicly available brain tumor MRI image dataset consisting of multiple tumor types and healthy brain scans, divided into:
- `train/`
- `test/`

Each class contains labeled images representing different tumor categories.

## Models Used & Accuracy
Three deep learning architectures were trained and evaluated:
| Model     | Accuracy (%) |
|-----------|--------------|
| DenseNet  | 91.7632%     |
| CNN       | 81.388%      |
| U-Net     | 77.421%      |

## Preprocessing Steps
- Image resizing to 128x128 pixels
- Grayscale conversion (if needed)
- Noise filtering and normalization (rescaling pixel values)
- Data augmentation: rotation, zoom, horizontal flip

## Training Details
- Batch Size: 16
- Epochs: 15
- Early Stopping with patience of 3
- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`

## Evaluation
The trained DenseNet model achieved the highest performance in terms of validation accuracy and generalization, making it the preferred choice for deployment.

## Future Scope
- Improve generalization across diverse datasets
- Incorporate clinical metadata for context-aware diagnosis
- Visualize model attention regions using Grad-CAM
- Deploy an end-to-end diagnostic system with a Streamlit interface
