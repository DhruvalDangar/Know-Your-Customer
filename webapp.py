import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

st.title("CUSTOMER CHURN PREDICTOR")
st.markdown("##### _Know your Customer!_ <hr>",True)

st.markdown("### Machine Learning Algorithm using Decision Tree Classifier ",True)

data = pd.read_csv('Bank_Customer_Churn_dataset.csv')
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

cat_cols = ['Geography', 'Gender']
remove = list()
for i in cat_cols:
    if (data[i].dtype == np.str or data[i].dtype == np.object):
        for j in data[i].unique():
            data[i+'_'+j] = np.where(data[i] == j,1,0)
        remove.append(i)
data = data.drop(remove, axis=1)

x=data.drop('Exited',axis=1)
y=data['Exited']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

gender = st.sidebar.selectbox("Select Gender",("Male","Female"))
GFemale = 0
GMale = 0
def gendervalue(gender):
    if gender == "Male":
        GMale = 1
        GFemale = 0
    else:
        GMale = 0
        GFemale = 1
gendervalue(gender)
a=st.sidebar.slider("Enter Age: ",18,92)
Country = st.sidebar.selectbox("Select Country",("France","Germany","Spain"))
GFrance = 0
GSpain = 0
GGermany = 0
def cont_name(Country):
    if Country == "France":
        GFrance = 1
        GSpain = 0
        GGermany = 0
    elif Country == "Germany":
        GFrance = 0
        GSpain = 0
        GGermany = 1
    else:
        GFrance = 0
        GSpain = 1
        GGermany = 0
cont_name(Country)
t=st.sidebar.slider("Enter Tenure: ",0,10)
crc = st.sidebar.selectbox("Do you have a Credit Card? ",("Yes","No"))
ccard = 0
def Creditcard(crc):
    if crc == "Yes":
        ccard = 1
    else:
        ccard = 0
Creditcard(crc)
act = st.sidebar.selectbox("Are you an active member? ",("Yes","No"))
mem = 0
def activemember(act):
    if act == "Yes":
        mem = 1
    else:
        mem = 0
activemember(act)
bal=st.sidebar.number_input("Enter Balance: ",0,250898,0,10000)
esal=st.sidebar.number_input("Enter Estimated Salary: ",12,199992,10000,1000)
cs=st.sidebar.number_input("Enter CreditScore: ",350,850,350,50)
nop=st.sidebar.selectbox("Enter the no. of Products: ",("1","2","3","4"))

import pickle

pickle.dump(model,open('finalmodel.pkl','wb'))
loadedmodel=pickle.load(open('finalmodel.pkl','rb'))
pred=loadedmodel.predict([[cs,a,t,bal,nop,ccard,mem,esal,GFrance,GSpain,GGermany,GMale,GFemale]])

if pred == 1:
    st.markdown("# Customer will LEAVE")   
else:
    st.markdown("# Customer will STAY")
