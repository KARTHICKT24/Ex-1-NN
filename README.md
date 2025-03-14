<H3>NAME : KARTHICK KISHORE T</H3>
<H3>REG NO : 212223220042</H3>.</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

NAME : KARTHICK KISHORE T
Reg no. : 212223220042

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import io

df=pd.read_csv('Churn_Modelling.csv')
df

x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated().sum()

df.drop(['Surname'],axis=1,inplace=True) 
df.drop(['CustomerId','Gender','Geography'],axis=1,inplace=True)
df

df.describe()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

x1=df1.iloc[:,:-1].values
print(x1)
y1=df1.iloc[:,-1].values
print(y1)

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))

## OUTPUT:

![nn1](https://github.com/user-attachments/assets/7680b420-ef64-4c31-b914-07441e0131cd)

![nn2](https://github.com/user-attachments/assets/e389095d-74dc-4c99-8417-4d4c4020b020)

![nn3](https://github.com/user-attachments/assets/ca0ab620-685d-4237-bd44-4013d6fcd1d5)

![nn4](https://github.com/user-attachments/assets/7e1bf6b2-dcfd-45d1-8938-5c4d37293dc9)

![nn5](https://github.com/user-attachments/assets/e0cec186-b6fc-4a8b-8f97-1a269b9919a2)

![nn6](https://github.com/user-attachments/assets/86d2a9d7-c19a-4719-a257-d4eafa3e58fd)

![nn8](https://github.com/user-attachments/assets/35162583-16d9-46d4-9a0b-3284874f0fb9)

![nn9](https://github.com/user-attachments/assets/d270a685-b90a-4d15-a8d1-a4a1ed678def)

![nn10](https://github.com/user-attachments/assets/d2a1dd14-4265-4d26-8c07-85c1c4c7878e)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


