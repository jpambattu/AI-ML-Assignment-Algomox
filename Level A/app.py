import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pickle

# loading pickle file for prediction
model = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv(r"C:\Users\Jpamb\Desktop\AIQ1\aml_eval.csv")  # read the dataset

X_test = df[[
    'amount',

    'PrevBalanceSender',
    'CurrentBalanceSender',

    'PrevBalanceReceiver',
    'CurrentBalanceReceiver']]

y_pred_rfe = model.predict(X_test)


prediction = pd.DataFrame(y_pred_rfe, columns=['IsSuspicious'])

# appending the prediction results to the evaluation datset
output_df = pd.concat([df, prediction], axis=1).to_csv(r"C:\Users\Jpamb\Desktop\AIQ1\aml_eval.csv")

features = list(map(float,input("\nEnter the numbers : ").strip().split()))[:5]
final_features = [np.array(features)]

df = pd.DataFrame(final_features)

pred = model.predict(df)

if int(pred[0]) == 1 :
    print("Not Suspicious")
else :
    print("Suspicious")


