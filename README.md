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

This neural network is compiled using the binary cross-entropy loss, the ADAM algorithm optimizer, and accuracy as an evaluated metric. A quick run fitting the model with the training data over fifty epochs shows that each epoch runs *804 batches*. This is used to calibrate the checkpoint system that sets up a callback which will store checkpoints into a separate folder. For the case of economy, only the weights of the model will be saved, and only every five epochs (which will require knowledge of the batch size to calibrate properly).

The trained model is then evaluated against the testing data with the loss and accuracy observed. The results for the final model and some arbitrary checkpoint are displayed below:

![Results of the First Neural Network](https://github.com/Owen-Wang1234/Neural_Network_Charity_Analysis/blob/main/Images/First_Neural_Network_Results.png)

## Optimization of the Neural Network
As seen in the figure above, the model is not the best it could be; the accuracy is **less than 75%** and the loss is **approximately 0.56** which are not ideal values. The next step is to seek ways to improve the performance of the neural network.

### Attempt 1: Adjust the Pre-Processing
One attempt to improve the performance involves adjusting the pre-processing. An examination of the features reveals that the `STATUS` and `SPECIAL_CONSIDERATIONS` features are very unbalanced in value distribution, and the minority classes did not appear to yield a significant impact. Thus, these two were also removed at the beginning.

The binning process was readjusted to allow more bins in the `APPLICATION_TYPE` and `CLASSIFICATION` features. Because this attempt focuses on the pre-processing, the neural network remains untouched.

The results from this attempt is illustrated below:
![Results of the First Attempt](https://github.com/Owen-Wang1234/Neural_Network_Charity_Analysis/blob/main/Images/Attempt1_Results.png)

The accuracy is still not at 75%, but the adjustments to the pre-processing has yielded some improvement.

### Attempt 2: Improve the Neural Network
The next attempt to improve the performance focuses on the neural network. One idea is to add more neurons to each layer and to try one more layer. The result is shown below:
![Second Neural Network Attempt](https://github.com/Owen-Wang1234/Neural_Network_Charity_Analysis/blob/main/Images/NeuralNet2.png)

The results from this attempt is illustrated here:
![Results of the Second Attempt](https://github.com/Owen-Wang1234/Nerual_Network_Charity_Analysis/blob/main/Images/Attempt2_Results.png)

The accuracy remains below 75%, and in fact has not changed from the previous attempt.

### Attempt 3: Adjust the Activation Functions
The last attempt involves the activation functions of the neural network. The layers that used the Rectified Linear Unit (ReLU) function changed over to the Hypberbolic Tangent (Tanh) function; the output layer stays on the Sigmoid function. The neural network is now:
![Third Neural Network Attempt](https://github.com/Owen-Wang1234/Neural_Network_Charity_Analysis/blob/main/Images/NeuralNet3.png)

The results from this attempt is illustrated here:
![Results of the Third Attempt](https://github.com/Owen-Wang1234/Neural_Network_Charity_Analysis/blob/main/Images/Attempt3_Results.png)

The accuracy still remains unchanged, and not at 75%.

### Special Note
During the attempts to optimize the neural network, it was noticed that even if the training and testing data sets stay unchanged, each compilation of the same neural network model yielded slightly different results. Each attempt with the same data and the same model resulted in varying accuracy percentages. This is attributed to the fact that the neural network cannot be fixed to one random state.

## Results
After three attempts, the neural network model still could not achieve the accuracy rating of 75%. Over these attempts:

- The primary target for the model is whether or not a charity succeeds with investor support (`IS_SUCCESSFUL`).
- The features removed from the DataFrame are considered non-factors. In addition to the `EIN` and `NAME` features, the `STATUS` and `SPECIAL_CONSIDERATIONS` features are also removed.
- The remaining features will be used in the model: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `INCOME_AMT`, `ASK_AMT`, and `IS_SUCCESSFUL`.

- The neural network model at the end contains four layers:
    1. The input layer contains 180 neurons, which is three times the number of input features.
    2. The next hidden layer contains 40 neurons, slightly more than the previous 30 but the calculated parameters do not exceed the number of parameters in the previous layer.
    3. The next hidden layer contains 30 neurons, slightly less than the number of neurons in the prior layer.
    4. The last layer is the output layer with only one neuron.
    - The non-output layers all used the Tanh activation function in place of the ReLU function since the ReLU function outputs zero for any negative input value, running the risk of "dying neurons" when many data points fall locked into zero.

- Despite three attempts, the neural network model cannot reach the accuracy of 75%. It always stagnated or fluctuated around the 73% mark.
- Attempts to improve the neural network model include:
    1. Adjusting the pre-processing to remove some features perceived as non-factors and allowing more bins to separate any data points that could have been confounding
    2. Expanding the neural network with more neurons and an extra hidden layer
    3. Trying different activation functions
    4. Attempts to increase the number of epochs yielded nothing productive as the accuracy plateaued somewhat quickly during training.

## Summary
The neural network model can provide a fairly okay performance with an accuracy between 72% and 73%, but none of the attempts at improvement described above have been able to improve the accuracy to at least 75%. This brings up the possibility of looking into other models that could serve as alternatives.

The neural network model, regardless of the number of layers and neurons, eventually ends with an output layer which contains only one neuron and runs the Sigmoid activation function. This output layer effectively serves as a logistic regression model, which is essentially based on the sigmoid curve. At this point, it may be worth looking into a logistic regression model, especially since the data has a relatively balanced distribution of outcomes (success vs. failure).

If the same pre-processing is used, then the logistic regression model can deliver at least comparable results that will be easier to follow and understand. A Random Forest Classifier model would be a solid candidate that can handle the large number of data points and various features reasonably well.