# Customer-Churn-Prediction-Using-ANN

This project involves predicting customer churn for a bank using an Artificial Neural Network (ANN). The dataset used is the Churn Modelling dataset, which contains various features about the customers and whether or not they exited (churned) from the bank. The model aims to classify customers based on these features and predict their likelihood of churning.

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation and Usage](#installation-and-usage)

## Dataset
The dataset contains the following columns:
- Customer demographics such as `Geography`, `Gender`, `Age`, etc.
- Account information such as `CreditScore`, `Balance`, `Tenure`, etc.
- A target variable `Exited` which indicates whether a customer churned or not (0 = No, 1 = Yes).

You can download the dataset [here](https://www.kaggle.com/datasets/shubhendra12/churn-modelling).

## Data Preprocessing
- **Handling Duplicates**: Checked for duplicate entries and removed them.
- **Dropped Irrelevant Columns**: Columns such as `RowNumber`, `CustomerId`, and `Surname` were dropped as they don't contribute to the prediction.
- **Encoding Categorical Variables**: `Geography` and `Gender` were converted into dummy variables using `pd.get_dummies()`.
- **Feature Scaling**: Used `StandardScaler` to standardize the features before feeding them into the neural network.

## Feature Engineering
After preprocessing, the dataset contains the following columns:
- Numerical features: `CreditScore`, `Age`, `Tenure`, `Balance`, etc.
- Categorical features: `Geography_France`, `Geography_Germany`, `Gender_Female`, etc.

The target variable is `Exited`.

## Model Architecture
An Artificial Neural Network (ANN) was used to build the classification model. The architecture consists of:
- **Input Layer**: 13 input features.
- **Two Hidden Layers**: Each with 5 neurons and ReLU activation function.
- **Output Layer**: A single output neuron with a sigmoid activation function to predict binary output (churn or not).

## Training and Evaluation
- The model was trained on 80% of the data using a `binary_crossentropy` loss function and the `Adam` optimizer. 
- A validation split of 20% was used during training. 
- The dataset was split into training and test sets using `train_test_split`, with 20% of the data held out for testing.

### Accuracy and Performance
After training, the model achieved an accuracy of around **85%** on the test set.

A confusion matrix was generated to visualize the classification performance.

## Results
- **Training Accuracy**: Approximately 85%
- **Validation Accuracy**: Consistently high, as plotted in the accuracy curve.
- **Confusion Matrix**: Visualized to analyze the prediction results and errors.

## Conclusion
The ANN model successfully predicts customer churn with an accuracy of **0.85**. The model can be further tuned by experimenting with different architectures, optimizers, or by applying techniques such as dropout or regularization to improve performance.

