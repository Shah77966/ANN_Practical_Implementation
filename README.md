# ANN Practical Implication

This README provides an overview and instructions for running the code snippet provided. The code implements an Artificial Neural Network (ANN) model using TensorFlow and scikit-learn libraries to perform classification on a dataset.

## Prerequisites

To run this code, you need to have the following libraries installed:

- scikit-learn
- pandas
- matplotlib
- numpy
- tensorflow

Please make sure you have these libraries installed before proceeding.

## Code Overview

The code performs the following steps:

1. Imports the required libraries:

   - `accuracy_score` and `confusion_matrix` from `sklearn.metrics`
   - `Dropout`, `LeakyReLU`, `PReLU`, `ReLU`, and `Dense` from `tensorflow.keras.layers`
   - `Sequential` from `tensorflow.keras.models`
   - `StandardScaler` from `sklearn.preprocessing`
   - `train_test_split` from `sklearn.model_selection`
   - `pandas` as `pd`
   - `matplotlib.pyplot` as `plt`
   - `numpy` as `np`
   - `tensorflow` as `tf`

2. Prints the TensorFlow version.

3. Loads the dataset from the "Churn_Modelling.csv" file.

4. Divides the dataset into independent features (`x`) and the target variable (`y`).

5. Performs feature engineering by encoding categorical variables ("Geography" and "Gender") using one-hot encoding.

6. Concatenates the encoded features with the original dataset.

7. Splits the dataset into training and test sets.

8. Performs feature scaling on the training and test sets using `StandardScaler`.

9. Creates an instance of the ANN model using `Sequential`.

10. Adds the input layer with 11 nodes and the activation function "relu".

11. Adds two hidden layers with 7 and 6 neurons, respectively, using the "relu" activation function.

12. Adds a dropout layer with a dropout rate of 0.2.

13. Adds the output layer with 1 neuron and the activation function "sigmoid".

14. Compiles the model using the Adam optimizer with a learning rate of 0.001, binary cross-entropy loss function, and accuracy as the evaluation metric.

15. Defines early stopping as a callback to stop training if the validation loss does not improve for a certain number of epochs.

16. Trains the model on the training data with a validation split of 0.33 and a batch size of 10, using early stopping.

17. Plots the model's accuracy and loss history during training.

18. Performs predictions on the test set and converts the predictions to binary values (0 or 1) using a threshold of 0.5.

19. Calculates the confusion matrix and accuracy score based on the predicted values and the actual test labels.

20. Prints the confusion matrix and the accuracy score.

21. Prints the weights of the trained model.

## Running the Code

To run the code:

1. Make sure you have the required libraries installed.

2. Download the "Churn_Modelling.csv" dataset and place it in the same directory as the code file.

3. Run the code snippet.

Upon running the code, you will see the TensorFlow version, the training progress with the validation loss, and the accuracy and loss plots. Finally, the confusion matrix and accuracy score will be printed.

Please note that you may need to modify certain parts of the code to suit your specific dataset or requirements.

That's it! You should now have a good understanding of the code and how to run it. If you have any further questions, feel free to ask.
