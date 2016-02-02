# Baye-s-Theorem
This example is to solve the classification problem and compute the training and testing errors.

## Code analysis of part 1
- Initially we read the data from the file row by row and saved each row into
the list and finally added all the list into one list by using the user defined
function ’read_file(filename)’.
- Then we calculated the mean and the standard deviation of all the 12 fea-
tures by using the numpy build in functions ’mean(x)’ and ’std(x)’ and saved
them to an list and returned by the function ’cal_mean_std(data_from_file)’.
- Counted the number of class 1 and class 2 by iterating through the labels
from the data file and computed the probability of class 1 and class 2. For
calculating the probability used the user defined function
’cal_class_probability(data_from_file and class)’.
- Calculated the posterior probability i.e P(X | H) using the function
’cal_posteriori_each_feature(data,mean,std)’.
- Now computed the priori probability of the features combined i.e P(X),
using the function ’cal_priori_features(probabilities)’.
- Calculated the posterior probability of the particular feature of particular
class using the function ’cal_posteriori_class(probabilities)’.
- Now we have all the data to implement equation[1] and calculate the prob-
ability of particular class when a feature is given P(H | X).
- Now we classified the data into class 1 or class 2 by comparing the poste-
rior probabilities calculated for class 1 and class 2. This is done using the
function ’create_expected_labels(probabilities)’.
- After the getting the ground truth labels by the function ’create_labels(data)’,
we compared with the predicted labels to find the number of matches and
mismatches.
- Error is obtained by taking number of mismatches into consideration.

## Code analysis of part 2
- In this part we had used most of the functions of the part 1 for classification.
Initially we had taken the all the specified percentages into a list.
- We had split the data into training and testing using the function ’split_data(data)’.
We used the same process as in part 1 to compute the training error and cal-
culated the count of labels to get the confusion matrix and printed to the
console.
- We repeated the process for all the percentages of training data and com-
puted the training and testing errors. For this we used to function ’com-
pute_fun(spliteddata)’.
- All the testing and the training errors generated are written to the data file
’Generated_error_table.dat’.
- By reading the generated data file we have plotted the training and the test-
ing error on the same plot.

