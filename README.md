# NYC Rolling Sales Prediction using TensorFlow

This repository contains an implementation of a neural network model using TensorFlow to predict NYC rolling sales prices. The code preprocesses the dataset, trains a regression model, and evaluates the model's performance using various metrics.

## Dataset

The dataset used for this prediction task is the NYC Rolling Sales dataset, which includes information about various properties sold in New York City. The dataset contains features such as the borough, neighborhood, building class category, and sale price.

You can download the dataset from the following source:
[NYC Rolling Sales Dataset](https://www.kaggle.com/new-york-city/nyc-rolling-sales)

## Requirements

- pandas
- numpy
- scikit-learn
- tensorflow

You can install the required packages by running the following command:
pip install pandas numpy scikit-learn tensorflow


## Usage

1. Clone the repository or download the `nyc_sales_prediction.py` file.

2. Make sure you have the NYC Rolling Sales dataset (`nyc-rolling-sales.csv`) in the correct directory. Adjust the file path in the code if necessary.

3. Run the script using a Python interpreter:
python nyc_sales_prediction.py


4. The script will preprocess the dataset by applying label encoding to convert categorical features into numeric values. It will then split the data into training and test sets using a 50:50 ratio.

5. Next, the script will define and train a neural network model using TensorFlow. The model consists of a single dense layer with ReLU activation.

6. The model will be compiled with the Mean Squared Error (MSE) loss function and the Adam optimizer. It will then be trained on the training set for 30 epochs.

7. After training, the model will make predictions on the test set, and various evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Explained Variance Score will be calculated.

8. A scatter plot will be displayed, showing the predicted sale prices versus the actual sale prices from the test set. The plot will also include a reference line for perfect predictions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
