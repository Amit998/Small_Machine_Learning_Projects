# import pandas as pd
# import csv
# # load data
# data=pd.read_csv("data/owid-covid-data.csv",sep=',')
# df=pd.DataFrame(data)


# indiaData=df.loc[df['location'] == 'India']
# # print([indiaData])

# Date=list(indiaData['date'])
# Day=1
# total_cases=list(indiaData['total_cases'])
# new_cases=list(indiaData['new_cases'])

# total_deaths=list(indiaData['total_deaths'])
# new_deaths=list(indiaData['new_deaths'])

# new_cases_per_million=list(indiaData['new_cases_per_million'])

# new_tests=list(indiaData['new_tests'])
# new_cases=list(indiaData['new_cases'])
# total_tests=list(indiaData['total_tests'])
# tests_per_case=list(indiaData['tests_per_case'])
# positive_rate=list(indiaData['positive_rate'])
# hospital_beds_per_thousand=list(indiaData['hospital_beds_per_thousand'])
# population=list(indiaData['population'])



# mainList=[]

# # print(Date)
# for j in range(len(Date)):
#     mainList.append([Day,Date[j],int(total_cases[j]),new_cases[j],total_deaths[j],new_deaths[j],new_cases_per_million[j],new_tests[j],new_cases[j],total_tests[j],tests_per_case[j],positive_rate[j],hospital_beds_per_thousand[j],population[j]])
#     Day+=1
    
    
# # indiaTotalCase=indiaData['total_cases']
# # print(indiaTotalCase)


# print(mainList)

# filename="covid19India.csv"
# fields=["Day","Date","total_cases","new_cases","total_deaths","new_deaths","new_cases_per_million","new_tests","new_cases","total_tests","tests_per_case","positive_rate","hospital_beds_per_thousand","population"]
# # # # i=[[count]]
# with open(filename,'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     for i in indiaDate:
#         tempList=[i,counter]
        
#         csvwriter.writerows(tempList)
    


#     # writing the data rows  
#     csvwriter.writerows(i)
#     i+=1



# data=data['location']='India'
# print(data.head())

# print(len(mainList))

# import csv
# with open('covid19India.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(fields)
#     # for i in range(len(mainList)):
#         # print(mainList[i])
#     writer.writerows(mainList)





import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as  plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

data=pd.read_csv('covid19India.csv',sep=',')
data=data[['Day','total_cases']]

print('-'*30);print('Head');print('*'*30)
print(data.head())

# Print Data Header

print('-'*30);print('Prepare Data');print('*'*30)


x=np.array(data['Day']).reshape(-1,1)
y=np.array(data['total_cases']).reshape(-1,1)


# for i in y:
#     print(type(i))


print(data.isnull().sum())
print(len(data))
data=data.dropna()
print(data.isnull().sum())
print(type(data))







polyFeat=PolynomialFeatures(degree=5)
x=polyFeat.fit_transform(x)
# print(x)


# Tranining Data Header
print('-'*30);print('Prepare Data');print('*'*30)


model=linear_model.LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y)
print(f'Accuracy',round(accuracy*100,3),'%')
y0=model.predict(x)


# plt.plot(y,'r*')
# plt.plot(y0,'--b')
# plt.show()


# Prediction

days=30
print('-'*30);print('Prediction Data');print('*'*30)

print(f'Prediction After {days} Days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[int(len(data["Day"]))+days]]))),0),'Milion')

x1=np.array(list(range(1,int(len(data["Day"]))+days))).reshape(-1,1)
y1=model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'r-')

plt.plot(y,'r*')
plt.plot(y0,'--b')
plt.show()