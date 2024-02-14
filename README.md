# Fashion
# Fashion MNIST Image Classification using TensorFlow and Keras

This project demonstrates the implementation of a fashion image classification model using TensorFlow and Keras. The model is trained on the Fashion MNIST dataset to classify clothing images into 10 categories.

## Dataset

The Fashion MNIST dataset consists of 60,000 training images and 10,000 testing images of clothing items across 10 different categories. Each image is a grayscale image of size 28x28 pixels.

## Installation

To run the code locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fashion-mnist-classification.git
   1. Navigate to the project directory:
      cd fashion-mnist-classification
   2. Install the required dependencies using pip:
       pip install -r requirements.txt
   3. Run the main script:
      python main.py
Code Structure
- `main.py`: Main script to load and preprocess the Fashion MNIST dataset, define the model architecture, train the model, and evaluate its performance.
- `utils.py`: Utility functions for data loading, preprocessing, and plotting.
- `requirements.txt`: List of Python dependencies required to run the project.

Model Architecture
The model architecture consists of the following layers:

- Flatten Layer: Flattens the input image from a 28x28 matrix to a 1D array.
- Dense Layer: Fully connected layer with 128 neurons and ReLU activation function.
- Dense Layer: Output layer with 10 neurons (equal to the number of classes) and no activation function.

Training
The model is trained using the Adam optimizer and Sparse Categorical Crossentropy loss function. The training process involves 10 epochs with a batch size of 32.

Evaluation
The trained model achieved an accuracy of approximately 87.3% on the test set. Additionally, a classification report and confusion matrix are provided to evaluate the model's performance across different classes.

Results
- Training Accuracy: 91.10%
- Test Accuracy: 87.29%

Conclusion
This project demonstrates the implementation of a simple image classification model using TensorFlow and Keras. The model performs reasonably well in classifying fashion images into different categories.

Feel free to experiment with different model architectures, optimizers, and hyperparameters to further improve the performance.
