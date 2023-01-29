# Neural_Network_Charity_Analysis

## Project Overview
The client is in the business of providing financial support to various charities. The primary challenge for the client is to determine which charity is likely to succeed with investment aid, thus meriting attention, as opposed to one that would fail even with support. So the client requests a method to reliably predict which charities merit investment among a very large volume of applicants based on various traits. THe plan is to develop a binary classifier to evaluate the applicant pool using neural networks.

## Resources

### Data Sources

- charity_data.csv
- AlphabetSoupCharity.ipynb

### Software
The unsupervised machine learning model is developed in Python within the machine learning environment (mlenv):

- Python 3.7.15
- Jupyter Notebook 6.5.2
- Pandas 1.3.5
- scikit-learn 1.0.2
- tensorflow 2.11.0

## Pre-Process the Data
When the data is imported into a DataFrame, the first glance shows that the ID and the name of the applicant will not be needed when training the neural network, so the two columns are dropped. Checking the remaining columns, three of them have more than ten unique values, and the `ASK_AMT` column is numeric type while the other two are categorical types. For both categorical columns (`APPLICATION_TYPE` and `CLASSIFICATION`) a count of unique values is gathered and plotted in a density chart. Looking at both the counts and the chart, the values with counts less than some defined number are listed; every instance of these values in the column is replaced with "Other" as part of the binning process for less common category variables.

After the binning process, the columns with categorical types are listed to be used in the fitting and transformation of the applicant DataFrame under the `OneHotEncoder` method. The results are merged into the applicant DataFrame while dropping the original categorical columns.

Whether the applicant is successful or not is the target parameter, so all the other columns are features, and the `X` and `y` arrays are established accordingly. These arrays are divided into training and testing sets using the random state of one. The features (training and testing) are standardized using the `StandardScaler` fitted with the training data.

## First Attempt with TensorFlow Neural Network
After the training and testing data sets are prepared, a neural network is created to set up a binary classification model. The first one tested is a deep neural network with two hidden layers. The number of input features is quickly calculated by extracting the first row of the training X matrix and getting the length. The first hidden layer takes the input features and runs eighty nodes; the features matrix holds 43 columns, so a general convention is to use about twice to three times as many nodes. The first hidden layer designates the Rectified Linear Unit (ReLU) function as the activation function. The second hidden layer runs thirty nodes and also designates the ReLU function as the activation function. The last layer is the output layer with one single node and uses the Sigmoid function as the activation function.

This neural network is compiled using the binary cross-entropy loss, the ADAM algorithm optimizer, and accuracy as an evaluated metric. A quick run fitting the model with the training data over fifty epochs shows that each epoch runs 804 batches. This is used to calibrate the checkpoint system that sets up a callback which will store checkpoints into a separate folder. For the case of economy, only the weights of the model will be saved, and only every five epochs (which will require knowledge of the batch size to calibrate properly).

The trained model is then evaluated against the testing data with the loss and accuracy observed. The results for the final model and some arbitrary checkpoint are displayed below:

![Results of the First Neural Network](https://www.github.com/Owen-Wang1234/Neural_Network_Charity_Analysis/blob/main/First_Neural_Network_Results.png)