import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import warnings
import csv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

dataset=pd.read_csv("healthcare-dataset-stroke-data.csv")
data1=dataset.set_index("id")
data2=dataset.set_index("id")
data3=dataset.set_index("id")
data4=dataset.set_index("id")
bmi_mean=format(data2["bmi"].mean(),'.1f')
mbfound = False
ssfound= False
#dataset 1
for rec in range(0,len(data1)):
    if(not mbfound):
        if(mt.isnan(data1["bmi"][dataset["id"][rec]])):
            data1=data1.drop('bmi',1)
            mbfound=True
    if(not ssfound):
        if(data1["smoking_status"][dataset["id"][rec]]=="Unknown" and not ssfound):
            data1=data1.drop('smoking_status',1)
            ssfound=True

#dataset 2
for rec in range(0,len(data2)):
    if(mt.isnan(data2["bmi"][dataset["id"][rec]])):
        data2["bmi"][dataset["id"][rec]]=bmi_mean

#dataset 3

zcount=0

for rec in range(0,len(data3)):
    if(mt.isnan(dataset["bmi"][rec])):
        row=dataset["id"][rec]
        zcount=zcount+1
        idx = [data3.index[i] for i in range(len(data3)) if data3.index[i] != row] + [row]
        data3=data3.reindex(idx)

tsize=zcount/len(data3)
print(tsize)
x = data3[['avg_glucose_level']]
y=data3['bmi']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=tsize, shuffle=False)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
for i in range(0,len(data3)):
    if(data3.index[i] in df.index):
        data3['bmi'][data3.index[i]]=format(df['Predicted'][data3.index[i]],'.1f')
data3=data3.drop('smoking_status',1)
#dataset 4-5

for i in range(0,len(data4)):
    if(mt.isnan(data4["bmi"][data4.index[i]])):
        data4["bmi"][data4.index[i]]=data3["bmi"][data4.index[i]]

zcount=0

for rec in range(0,len(data4)):
    if(dataset["smoking_status"][rec]=='Unknown'):
        row=dataset['id'][rec]
        zcount=zcount+1
        idx1 = [data4.index[i] for i in range(len(data4)) if data4.index[i] != row] + [row]
        data4=data4.reindex(idx1)
#data4.to_csv (r'export_dataframe.csv', index = False, header=True)

print(len(data4))
print(zcount)
tsize=zcount/len(data4)
print(tsize)
x = data4.iloc[:,[1,7,8]].values
y=data4['smoking_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=tsize, shuffle=False)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
count=0

for i in range(0,len(data4)):
    if(data4.index[i] in df.index):
        data4['smoking_status'][data4.index[i]]=df['Predicted'][data4.index[i]]
data5=data4
data4=data4.drop('bmi',1)
#Random Forest for data1
data1=pd.get_dummies(data1)
y=data1['stroke']
x = data1.drop('stroke',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=1)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print(" Classification Report for dataset 1")
print(classification_report(y_test.iloc[:].values, y_pred))
print('\n')

#Random Forest for data2
data2=pd.get_dummies(data2)
y=data2['stroke']
x = data2.drop('stroke',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=1)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print(" Classification Report for dataset 2")
print(classification_report(y_test.iloc[:].values, y_pred))
print('\n')

#Random Forest for data3
data3=pd.get_dummies(data3)
y=data3['stroke']
x = data3.drop('stroke',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=1)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print(" Classification Report for dataset 3")
print(classification_report(y_test.iloc[:].values, y_pred))
print('\n')

#Random Forest for data4
data4=pd.get_dummies(data4)
y=data4['stroke']
x = data4.drop('stroke',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=1)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
count=0
for i in range(0,len(y_pred)):
    if(y_test.iloc[:].values[i]==y_pred[i]):
        count+=1
print(count/len(y_pred))

print(" Classification Report for dataset 4")
print(classification_report(y_test.iloc[:].values, y_pred))
print('\n')

#Random Forest for data5
data5=pd.get_dummies(data5)
y=data5['stroke']
x = data5.drop('stroke',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=1)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
#for i in range(0,len(y_pred)):
    #print({'Actual':y_test.iloc[:].values[i],'Predicted':y_pred[i]})

print(" Classification Report for dataset 5")
print(classification_report(y_test.iloc[:].values, y_pred))
print('\n')


