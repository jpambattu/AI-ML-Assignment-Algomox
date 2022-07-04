# Import all necessary libraries
import pandas as pd  # for analyzing data
import numpy as np  # mathmetical operation
from matplotlib import pyplot as plt # plotting confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

import pickle # produce model.pkl file

from sklearn.model_selection import train_test_split  # split data into train data set and test data set

# feature Selection
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer # data cleaning

from sklearn.preprocessing import StandardScaler # feature engineering

df = pd.read_csv(r"C:\Users\Jpamb\Desktop\AIQ1\aml_train.csv")  # read the dataset

# Used drop() function for removing unnecessary features
train_data = df.drop(['Unnamed: 14'], axis='columns', inplace=False)

# Used simpleImputer for imputing null values and missing data which return most frequent data and replace them.
impt = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data = impt.fit_transform(train_data)

train_data = pd.DataFrame(train_data,
                          columns=['TxnID', 'token', 'TransactionType', 'amount', 'SenderID', 'PrevBalanceSender',
                                   'CurrentBalanceSender', 'ReceiverID', 'PrevBalanceReceiver',
                                   'CurrentBalanceReceiver', 'time', 'Sender Location', 'Receiver Location',
                                   'IsSuspicious'])

# function to standardize the data values into a standard format; resize the distribution of values
standardScaler = StandardScaler()
columns_to_scale = ['amount',

                    'PrevBalanceSender',
                    'CurrentBalanceSender',

                    'PrevBalanceReceiver',
                    'CurrentBalanceReceiver']

train_data[columns_to_scale] = standardScaler.fit_transform(train_data[columns_to_scale])

# Flittering out the unnecessary columns & store the new dataset which is lbw_data_sub
train_data_sub = train_data[[
    'amount',

    'PrevBalanceSender',
    'CurrentBalanceSender',

    'PrevBalanceReceiver',
    'CurrentBalanceReceiver',
    'IsSuspicious']]

# segregating dataset into features i.e., X and target variables i.e., y
X = train_data_sub.drop(['IsSuspicious'], axis=1)
y = train_data_sub['IsSuspicious']

y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# model building
rf_ent = RandomForestClassifier(criterion='entropy', n_estimators=100)
rf_ent.fit(X_train, y_train)
y_pred_rfe = rf_ent.predict(X_test)

#Pickle is used in serializing and deserializing a Python object structure, converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.
pickle.dump(rf_ent, open('model.pkl','wb'))


# Since this is  classification problem, Confusion matrix is selected to understand the performance
CM=confusion_matrix(y_test,y_pred_rfe)

TN = CM[0][0] # True Negative
FN = CM[1][0] # False Negative
TP = CM[1][1] # True Positive
FP = CM[0][1] # False Positive
specificity = TN/(TN+FP)*100 # specificity
acc= accuracy_score(y_test, y_pred_rfe)*100 # accuracy
prec = precision_score(y_test, y_pred_rfe)*100 # precision
rec = recall_score(y_test, y_pred_rfe)*100 # sensitivity/recall
model_results =pd.DataFrame([['Random Forest',acc, prec,rec,specificity]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity'])

print(model_results)
print("Confusion matrix \n",CM)

file = open("accuracy_results.txt", "w")
file.write(str(model_results))
file.write("\nConfusion Matrix \n")
file.write(str(CM))
file.close()

# Print the confusion matrix using Matplotlib

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(CM, cmap=plt.cm.Blues, alpha=0.3)
for i in range(CM.shape[0]):
    for j in range(CM.shape[1]):
        ax.text(x=j, y=i, s=CM[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


