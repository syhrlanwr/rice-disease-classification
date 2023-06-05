# Rice Disease Classification

This project focuses on classifying rice leaf images into different disease categories using deep learning techniques. It utilizes the EfficientNetV2 pre-trained model as a feature extractor and builds a custom classifier on top of it.

## Description

Rice is one of the most important cereal crops worldwide, providing a staple food source for a large portion of the global population. However, various diseases can affect rice plants, leading to reduced crop yield and economic losses for farmers. Early detection and accurate classification of rice diseases can help mitigate these losses by enabling timely interventions and targeted treatments.

This project aims to develop an automated system for rice disease classification based on leaf images. By leveraging deep learning algorithms, the system can quickly analyze and classify images of rice leaves into different disease categories, such as BrownSpot, Healthy, Hispa, and LeafBlast. This allows farmers and agricultural experts to identify diseased plants and take appropriate measures to prevent further spread and minimize crop damage.

## Dataset

The dataset used for training and validation consists of rice leaf images with four disease categories: BrownSpot, Healthy, Hispa, and LeafBlast. The dataset is divided into training and validation sets using a split ratio of 90:10. The images are resized to a size of 224x224 pixels.

## Model Architecture

The model architecture consists of the EfficientNetV2 pre-trained model as the base feature extractor, followed by several fully connected layers. The final layer uses a softmax activation function for multi-class classification.

## Results

The model achieved the following results on the validation set:

|   Disease   | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| BrownSpot   |   0.82    |  0.94  |   0.88   |   53    |
| Healthy     |   0.87    |  0.84  |   0.85   |  149    |
| Hispa       |   0.69    |  0.74  |   0.71   |   57    |
| LeafBlast   |   0.94    |  0.86  |   0.90   |   78    |

- Accuracy: 0.84
- Macro Avg F1-Score: 0.84
- Weighted Avg F1-Score: 0.84

These results demonstrate the performance of the model in classifying rice leaf images into different disease categories.


## Dependencies

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Pillow
- Pandas
- NumPy
- scikit-learn

## Acknowledgments

- The rice leaf image dataset used in this project: [Rice Leaf Images](https://www.kaggle.com/datasets/nizorogbezuode/rice-leaf-images)
- TensorFlow for providing the EfficientNetV2 pre-trained model through TensorFlow Hub.
