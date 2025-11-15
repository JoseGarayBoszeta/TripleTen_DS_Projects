# Project: Age Detection using Computer Vision

## 1. Project Goal

The primary objective of this project is to develop a computer vision model capable of predicting a person's age from a photograph. This model is intended to be used by a supermarket chain to help verify customer age at self-checkout kiosks, particularly for the purchase of age-restricted items.

The model's performance will be evaluated against a predefined business metric to ensure it is more accurate than a simple baseline.

## 2. Data

The project utilizes a labeled dataset of facial images.
* **Images:** The dataset consists of thousands of photographs of faces.
* **Labels:** Each image is associated with a label corresponding to the person's real age.

## 3. Methodology

### A. Data Preparation
1.  **Loading Data:** The dataset of images and their corresponding age labels was loaded for processing.
2.  **Data Generators:** A Keras `ImageDataGenerator` was used to efficiently load images in batches.
3.  **Preprocessing:** All images were resized to a uniform dimension (e.g., 224x224 pixels) and pixel values were rescaled to be between 0 and 1.
4.  **Data Augmentation:** To help the model generalize better and prevent overfitting, data augmentation techniques (such as horizontal flipping) were applied to the training images.

### B. Model Development
1.  **Architecture:** A Convolutional Neural Network (CNN) was selected, as this architecture is the standard for image-based tasks.
2.  **Transfer Learning:** To leverage a powerful, pre-existing architecture, the **ResNet50** model (pre-trained on the ImageNet dataset) was used as the base.
3.  **Model Customization:** The final layers of the pre-trained model were replaced with a new "head" suitable for this specific regression task (predicting a continuous age value).
4.  **Training:** The model was trained using the Adam optimizer and Mean Squared Error (MSE) as the loss function, and its performance was monitored on a separate validation set.

## 4. Evaluation & Conclusion

The model was trained for a number of epochs until its performance on the validation set stabilized.

The final, trained model was then evaluated on a holdout **test set** (data the model had never seen) to get a final performance score. The primary metric used for this evaluation was **Mean Absolute Error (MAE)**, which measures the average number of years the model's prediction was off from the true age.

The project concluded by comparing this final MAE score to the business-defined baseline to determine if the model was successful.

## 5. Key Libraries and Tools
* **TensorFlow & Keras:** For building, training, and evaluating the CNN.
* **Pandas:** For loading and managing the age labels.
* **Seaborn & Matplotlib:** For initial data exploration.
