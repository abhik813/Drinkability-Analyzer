import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


water_data = pd.read_csv("./water_prediction.csv")

water_data_left = water_data.dropna()

plt.figure(figsize=(12,10))
cor = water_data_left.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

cor_target = abs(cor["Potability"])
r_f = cor_target[cor_target>0.1] 

water_data_left.hist(figsize = (11,10))
sns.countplot(x = water_data_left['Potability'])
from sklearn.preprocessing import StandardScaler

input_cols = list(water_data_left.columns)[:-1]

output_cols = ["Potability"]

from sklearn.preprocessing import MinMaxScaler

input_df = water_data_left[input_cols].copy()
output_df = water_data_left[output_cols].copy()
scaler = MinMaxScaler().fit(input_df[input_cols])
input_df[input_cols] = scaler.transform(input_df[input_cols])


from sklearn.model_selection import train_test_split

input_train,input_val,output_train,output_val = train_test_split(input_df,output_df,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(solver='newton-cg',random_state=42)
model1.fit(input_train,output_train)

pred = model1.predict(input_train)

from sklearn.metrics import accuracy_score
accuracy_score(output_train,pred)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=135,max_depth=13)
model.fit(input_train,output_train)

model.score(input_train,output_train)

model.score(input_val,output_val)

from xgboost import XGBClassifier

n_est = [10,12,16,20,30,40,60,80]
res = []
inp = []

for i in range(8):
     model3 = XGBClassifier(n_jobs=-1,random_state=42,n_estimators=n_est[i],max_depth=30)
     model3.fit(input_train,output_train)
     inp.append(model3.score(input_train,output_train))
     res.append(model3.score(input_val,output_val))


plt.plot(res)

model3.fit(input_train,output_train)

model3.score(input_train,output_train)

model3.score(input_val,output_val)
from sklearn.ensemble import VotingClassifier

y_pred_val = 0

from sklearn.metrics import accuracy_score
estimator = []
estimator.append(('LR',LogisticRegression(solver='liblinear')))
estimator.append(('RFC',RandomForestClassifier(random_state=42,n_jobs=-1,n_estimators=135,max_depth=13)))
estimator.append(('XGB',XGBClassifier(n_jobs=-1,random_state=42,n_estimators = 12,max_depth=30)))
vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
vot_soft.fit(input_train, output_train)
y_pred = vot_soft.predict(input_train)
score_train = accuracy_score(output_train, y_pred)
y_pred_val = vot_soft.predict(input_val)

score_val = accuracy_score(output_val,vot_soft.predict(input_val))

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(output_val,y_pred_val,normalize='true')
sns.heatmap(mat,annot=True)

def input(finalarray):
    final = {
    "ph" : finalarray[0][0],
    "Hardness"  : finalarray[0][1],
    "Solids" : finalarray[0][2],
    "Chloramines" : finalarray[0][3],
    "Sulfate" : finalarray[0][4],
    "Conductivity" : finalarray[0][5],
    "Organic_carbon": finalarray[0][6],
    "Trihalomethanes":finalarray[0][7],
    "Turbidity": finalarray[0][8]
    }

    new_input_df = pd.DataFrame([final])
    new_input_df[input_cols] = scaler.transform(new_input_df[input_cols])
    return new_input_df
