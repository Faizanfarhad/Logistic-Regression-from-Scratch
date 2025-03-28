import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logistic_func
from sklearn.feature_selection import SelectKBest, f_classif

#start program

df = pd.read_csv('/media/LAPCARE/Algo_Bulding_fromScratch/logistic_Regression/House Price India.csv')
file_path = '/media/LAPCARE/Algo_Bulding_fromScratch/logistic_Regression/House Price India.csv'
classification_encoder = LabelEncoder()
encode = LabelEncoder()


target = input('Enter the target : ' )

y_actual = df.iloc[:,-1]
X = df.iloc[:,:-1]
X =  X.drop_duplicates()
X = X.drop(columns=['id','Number of schools nearby', 'number of views', 'Date', 'grade of the house', 'Postal Code', 'Longitude', 'Lattitude', 'Distance from the airport'])
X = (X - X.min()) / (X.max() - X.min())
X = pd.get_dummies(X, dtype=int)
df[target] = (df[target] > 100000).astype(int)
y_actual = df[target]

x_train,x_test,y_train,y_test = train_test_split(X,y_actual,test_size=0.20,random_state=42)


y_true = np.array(y_actual)
y_test = np.array(y_test)



def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def logistic_regression(X,w,b):
    z = np.dot(X,w) + b
    return sigmoid(z) 


def train(X:pd.DataFrame,y:pd.DataFrame, learningRate = 0.01, epoc=500):
    m,n = X.shape
    X,_ = logistic_func.encode_categorical_columns(X)
    X = np.array(X)
    w =np.zeros(n)
    print(w)
    b = 0

    for i in range(epoc):
        z = np.dot(X,w) + b
        prediction = sigmoid(z)
        dw = 1/m * np.dot(X.T, (prediction - y))
        db =  1/m * np.sum(prediction - y)
        
        w = w - learningRate * dw
        b = b - learningRate * db
        
        if i % 100 == 0:
            print(f'Prediction after {i} itreation : {prediction}')
        
        epsilon = 1e-15
        cost = (-1/m) * np.sum(y * np.log(prediction + epsilon) + (1 - y) * np.log(1 -prediction + epsilon))
        if i % 100 == 0:
            print(f"Cost after {i} iteration : {cost}")
    print(f'weight after all the epoc : {w}\n bias after all the epoc : {b}')
    return w,b

try:

    w,b = train(x_train,y_train)
    y_pred = logistic_regression(x_test,w,b)
    print('Probability of y_test :- ',y_pred)
except Exception as e:
    print(f'Error in calling: {e}')
    
