import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from scipy.stats import skew
from sklearn.utils import shuffle
import pylab as pl
%matplotlib inline

K_floods_df = pd.read_csv('kerala.csv')
K_floods_df.head(3)

K_floods_df.describe().transpose()

#We want the data in numbers, therefore we will replace the yes/no in floods coloumn by 1/0
K_floods_df['FLOODS'].replace(['YES','NO'],[1,0],inplace=True)

K_floods_df.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, figsize=(9,9), title='Box Plot for each input variable')
plt.savefig('boxplot')
plt.show()

plt.scatter(K_floods_df.FLOODS, K_floods_df.JUL, color='green')
plt.title('Floods vs July Rainfall')  
plt.xlabel('Chance of Floods') 
plt.ylabel('July Rainfall')
plt.show()

corr = K_floods_df.corr()  # get correlation matrix
sns.heatmap(corr,vmax=1,annot=True)    # plot correlation matrix

K_floods_df.hist()
plt.show()

shuffled_data = shuffle(K_floods_df)
y=shuffled_data.FLOODS
x=shuffled_data.drop(['FLOODS','SUBDIVISION','YEAR',' ANNUAL RAINFALL'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.60, random_state = 0)
y_train.head(2)

from sklearn.neighbors import KNeighborsClassifier
 #KNN classifier is fitted to the dataset
KNN_f = KNeighborsClassifier()
my_KNN_f = KNN_f.fit(x_train, y_train)
y_pred = my_KNN_f.predict(x_test)
pd.DataFrame({'Actual':y_test, 'Predict': y_pred}).head()
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("F1 score=",f1_score(y_test, y_pred, average="macro"))
print("Precision score=",precision_score(y_test, y_pred, average="macro"))
print("Recall Score=",recall_score(y_test, y_pred, average="macro"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("F1 score=",f1_score(y_test, y_pred, average="macro"))
print("Precision score=",precision_score(y_test, y_pred, average="macro"))
print("Recall Score=",recall_score(y_test, y_pred, average="macro"))

pd.DataFrame({'Actual':y_test, 'Predict': y_pred}).head()

from sklearn.naive_bayes import GaussianNB
#naive bayes is fitted to the dataset
gnb = GaussianNB()
my_nb = gnb.fit(x_train, y_train)
y_pred = my_nb.predict(x_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("F1 score=",f1_score(y_test, y_pred, average="macro"))
print("Precision score=",precision_score(y_test, y_pred, average="macro"))
print("Recall Score=",recall_score(y_test, y_pred, average="macro"))

from sklearn.tree import DecisionTreeClassifier
#DecisionTreeClassifier is fitted to the dataset
dtc = DecisionTreeClassifier()
my_dtc = dtc.fit(x_train, y_train)
y_pred = my_dtc.predict(x_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("F1 score=",f1_score(y_test, y_pred, average="micro"))
print("Precision score=",precision_score(y_test, y_pred, average="macro"))
print("Recall Score=",recall_score(y_test, y_pred, average="macro"))

from sklearn.ensemble import RandomForestClassifier
rmf = RandomForestClassifier(max_depth=3,random_state=0)
rmf_clf = rmf.fit(x_train,y_train)
y_pred = rmf_clf.predict(x_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("F1 score=",f1_score(y_test, y_pred, average="macro"))
print("Precision score=",precision_score(y_test, y_pred, average="macro"))
print("Recall Score=",recall_score(y_test, y_pred, average="macro"))

PAK_floods_df = pd.read_csv('Rainfall_1901_2016_PAK.csv')
PAK_floods_df.head(3)

PAK_floods_df= PAK_floods_df.pivot(index=' Year', columns='Month', values='Rainfall - (MM)')

PAK_floods_df = PAK_floods_df.rename(columns = {"April":"APR","August":"AUG",'December':"DEC",'February':"FEB",'January':"JAN",'July':"JUL", 'June':"JUN",'March':"MAR",'May':"MAY",'November':"NOV",'October':"OCT",'September':"SEP"})
PAK_floods_df.columns

cols = PAK_floods_df.columns.tolist()
columnsTitles = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
PAK_floods_df= PAK_floods_df.reindex(columns=columnsTitles)
PAK_floods_df.head(3)

PAK_floods_pred_KNN = my_KNN_f.predict(PAK_floods_df)
y_pred_KNN