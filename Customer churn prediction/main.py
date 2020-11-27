import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


df=pd.read_csv("Customer-Churn.csv")
# print(df.head())

df.drop('customerID',axis='columns',inplace=True)

# print(df.dtypes)

# print(df.TotalCharges.values)
# print(df.MonthlyCharges.values)

# print(pd.to_numeric(df.TotalCharges,errors='coerce').isnull())
# df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]
# pd.to_numeric(df.TotalCharges)
# print(df.shape)

df1=df[df.TotalCharges!=" "]
# print(df1.shape)
pd.to_numeric(df1.TotalCharges)

df1.TotalCharges=pd.to_numeric(df1.TotalCharges)



# print(df1.TotalCharges)
# print(df1[df1.Churn=='No'].tenure)
# tenure_churn_no=df1[df1.Churn=='No'].tenure
# tenure_churn_yes=df1[df1.Churn=='Yes'].tenure

# plt.xlabel("tenure")
# plt.ylabel("Number Of Customers")
# plt.hist([tenure_churn_yes,tenure_churn_no],color=['green','red'],label=['Churn=Yes','Churn=No'])
# plt.legend()
# plt.show()





# mc_churn_no=df1[df1.Churn=='No'].MonthlyCharges
# mc_churn_yes=df1[df1.Churn=='Yes'].MonthlyCharges

# plt.xlabel("Monthly Charges")
# plt.ylabel("Number Of Customers")

# blood_sugar_men=[113,85,90,150,149,88,93,115,135,80,77,129]
# blood_sugar_women=[67,98,89,120,133,150,84,69,89,79,120,112,100] 

# plt.hist([mc_churn_yes,mc_churn_no],rwidth=0.95,color=['green','red'],label=['Churn=Yes','Churn=No'])
# plt.legend()
# plt.show()



def print_unique_col_value(df):
    for col in df:
        if(df[col].dtypes=='object'):
            print(f'{col} : {df[col].unique()}') 


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)
# df1.replace('No','0',inplace=True)
# df1.replace('Yes','1',inplace=True)
yes_no_columns=['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
# print_unique_col_value(df1)

for col in yes_no_columns:
    df1.replace({'Yes':1,'No':0},inplace=True)

# print_unique_col_value(df1)

# for col in df1:
#      print(f'{col} : {df1[col].unique()}') 

df1['gender'].replace({'Male':1,'Female':0},inplace=True)


# print(df1['gender'].unique())

df2=pd.get_dummies(data=df1,columns=['InternetService','Contract','PaymentMethod'])


# pd.get_dummies(data=df1,columns=['InternetService'])

# for col in df2:
#      print(f'{col} : {df2[col].unique()}') 

# print(df2.dtypes)

scale_column=['tenure','MonthlyCharges','TotalCharges']


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df2[scale_column]=scaler.fit_transform(df2[scale_column])


# for col in df2:
#      print(f'{col} : {df2[col].unique()}') 


X=df2.drop('Churn',axis='columns')
y=df2['Churn']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

# print(x_train.shape,y_train.shape)

import tensorflow as tf
from tensorflow import keras


model=keras.Sequential(
    [
        keras.layers.Dense(20,input_shape=(26,),activation='relu'),
        keras.layers.Dense(15,activation='relu'),
        keras.layers.Dense(1,activation='sigmoid'),
    ]
) 

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=100)
# print(model.summary())


yp=model.predict(x_test)


# print(y_test[:5])
# print(yp[:5])


y_pred=[]

for element in yp:
    if (element > 0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)


# print(y_test[:5])
# print(y_pred[:5])


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,y_pred))


import seaborn as sn

cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()