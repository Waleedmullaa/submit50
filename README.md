import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images),
        np.array(labels),
        test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category,
    numbered 0 through NUM_CATEGORIES - 1. Inside each category is
    some number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted
    as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3.
    `labels` should be a list of integer labels, representing the
    categories for each of the corresponding `images`.
    """

    images = []
    labels = []

    # Go through every traffic sign category
    for category in range(NUM_CATEGORIES):

        category_path = os.path.join(data_dir, str(category))

        # Go through every image in this category
        for filename in os.listdir(category_path):

            image_path = os.path.join(category_path, filename)

            # Read image
            image = cv2.imread(image_path)

            # Skip anything OpenCV cannot read
            if image is None:
                continue

            # Resize image to the required dimensions
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Store image and its category
            images.append(image)
            labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model.
    Assume that the input to the neural network will be of the shape
    (IMG_WIDTH, IMG_HEIGHT, 3).
    The output layer should have NUM_CATEGORIES units.
    """

    model = tf.keras.models.Sequential([

        # Input image
        tf.keras.layers.Input(
            shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # First convolutional layer
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        # Reduce image dimensions
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Second convolutional layer
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        # Reduce dimensions again
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convert feature maps into one-dimensional data
        tf.keras.layers.Flatten(),

        # Hidden layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Reduce overfitting
        tf.keras.layers.Dropout(0.5),

        # Output layer: one unit for every traffic sign category
        tf.keras.layers.Dense(
            NUM_CATEGORIES,
            activation="softmax"
        )
    ])

    # Compile neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()

  # Traffic

For this project, I experimented with different convolutional neural
network structures to classify traffic signs. I first considered using
a simple network with only one convolutional layer and one pooling
layer. This worked, but the model did not seem to capture enough of the
details needed to distinguish between similar traffic signs.

I then used two convolutional layers. The first convolutional layer
uses 32 filters and the second uses 64 filters. Each convolutional
layer is followed by a max-pooling layer to reduce the size of the
feature maps while keeping important image features. This performed
better because the second convolutional layer could learn more complex
patterns from the features detected by the first layer.

After the convolutional layers, I flattened the data and used a dense
hidden layer with 128 units. I also added dropout with a rate of 0.5.
Without dropout, the model could become too dependent on the training
data, so dropout helped reduce overfitting.

The final output layer has 43 units, one for each traffic sign
category, and uses softmax activation. I used the Adam optimizer and
categorical crossentropy as the loss function. Overall, the model gave
good accuracy while remaining relatively simple and fast to train.  
