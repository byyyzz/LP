import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")
df.head()

print (df.info())

#Kategorik de�erlerin g�sterimi
dtype_object=df.select_dtypes(include=['object'])
dtype_object.head()
for x in dtype_object.columns:
    print("{} unique values:".format(x),df[x].unique())
    print("*"*20)

print (df.isnull().sum())

np.sum(df.isna())

#Bo� olan sat�rlar�n silinmesi
df= df.dropna(axis=0)
print (df.isnull().sum())
print (df.shape)

df.to_csv(r'/users/Acer/mlfinal.csv',index=False)

#Ya� da��l�m� grafi�i
fig = plt.figure(figsize=(13,5))
colors=["orange","yellow"]
df.groupby("age").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("Ya�")
plt.ylabel("Say�lar")
plt.tight_layout()
plt.show()

#Hasta olanlar�n ya� da��l�m� grafi�i
fig = plt.figure(figsize=(13,5))
colors=["orange","yellow"]
df[df["is_patient"]==1].groupby("age").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("Ya�")
plt.ylabel("Say�lar")
plt.tight_layout()
plt.show()

#Hasta olmayanlar�n ya� da��l�m� grafi�i
fig = plt.figure(figsize=(13,5))
colors=["orange","yellow"]
df[df["is_patient"]==2].groupby("age").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("Ya�")
plt.ylabel("Say�lar")
plt.tight_layout()
plt.show()

#Cinsiyet da��l�m� grafi�i
colors=["pink","blue"]
df.groupby("gender").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("cinsiyet")
plt.ylabel("Say�lar")
plt.show()

#Hasta olanlar�n cinsiyet da��l�m� grafi�i
colors=["pink","blue"]
df[df["is_patient"]==1].groupby("gender").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Say�lar")
plt.tight_layout()
plt.show()

#Hasta olmayanlar�n cinsiyet da��l�m� grafi�i
colors=["pink","blue"]
df[df["is_patient"]==2].groupby("gender").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Say�lar")
plt.tight_layout()
plt.show()

#Hasta olanlar ve hasta olmayanlar grafi�i
colors=["red","green"]
df.groupby("is_patient").size().plot(kind='bar',color=colors)
plt.xticks(rotation=0)
plt.xlabel("Hastal�k")
plt.ylabel("Say�lar")
plt.show()

#veriyi n�merik hale getirme
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dtype_object=df.select_dtypes(include=['object'])
print (dtype_object.head())
for x in dtype_object.columns:
    df[x]=le.fit_transform(df[x])

print (df.head())

#verinin test,e�itim diye ayr�lmas� ve �l�eklendirilmesi
X = df.iloc[:,:10].values
y = df["is_patient"].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 20)





from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

score=[]
algorithms=[]


#KNN algoritmas�
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn.predict(X_test)
score.append(knn.score(X_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(X_test,y_test)*100)


from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()

from sklearn.metrics import classification_report

target_names=["1","2"]
print(classification_report(y_true, y_pred, target_names=target_names))

#NB algoritmas�
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)

score.append(nb.score(X_test,y_test)*100)
algorithms.append("Navie-Bayes")
print("Navie Bayes accuracy =",nb.score(X_test,y_test)*100)


from sklearn.metrics import confusion_matrix
y_pred=nb.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Navie Bayes Confusion Matrix")
plt.show()
target_names=["1","2"]
print(classification_report(y_true, y_pred, target_names=target_names))

#DecisionTree algoritmas�
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("Decision Tree accuracy:",dt.score(X_test,y_test)*100)
score.append(dt.score(X_test,y_test)*100)
algorithms.append("Decision Tree")


from sklearn.metrics import confusion_matrix
y_pred=dt.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
plt.show()
target_names=["1","2"]
print(classification_report(y_true, y_pred, target_names=target_names))

#LR algoritmas�
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
score.append(lr.score(X_test,y_test)*100)
algorithms.append("Logistic Regression")
print("Logistic Regression accuracy {}".format(lr.score(X_test,y_test)*100))


from sklearn.metrics import confusion_matrix
y_pred=lr.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
target_names=["1","2"]
print(classification_report(y_true, y_pred, target_names=target_names))

#SVM algoritmas�
from sklearn.svm import SVC
svm=SVC(random_state=10)
svm.fit(X_train,y_train)
score.append(svm.score(X_test,y_test)*100)
algorithms.append("Support Vector Machine")
print("svm test accuracy =",svm.score(X_test,y_test)*100)


from sklearn.metrics import confusion_matrix
y_pred=svm.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
plt.show()
target_names=["1","2"]
print(classification_report(y_true, y_pred, target_names=target_names))

#ANN algoritmas�
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X = df.iloc[:,:10].values
print (X.shape[0])
y = df['is_patient'].values.reshape(X.shape[0], 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
sc.fit(X_test)
X_test = sc.transform(X_test)

sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.02, max_iter=100)
sknet.fit(X_train, y_train)

score.append(sknet.score(X_test,y_test)*100)
algorithms.append("Artificial Neural Networks")
print("Ann test accuracy =",svm.score(X_test,y_test)*100)

y_pred = sknet.predict(X_test)
y_true=y_test

cm=confusion_matrix(y_true,y_pred)

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Artificial Neural Networks Confusion Matrix")
plt.show()
target_names=["1","2"]
print(classification_report(y_true, y_pred, target_names=target_names))

#Algoritmalar�n kar��la�t�r�lmas�
print (algorithms)
print (score)

x_pos = [i for i, _ in enumerate(algorithms)]

plt.bar(x_pos, score, color='orange')
plt.xlabel("Algoritmalar")
plt.ylabel("Basari Yuzdeleri")
plt.title("Basari Siralamalar")

plt.xticks(x_pos, algorithms,rotation=90)

plt.show()
