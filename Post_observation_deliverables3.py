

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ClassificationAlgorithm import ClassificationAlgorithms


data = pd.read_csv('return_rates.csv')
data
data.columns


#----------------------------------------------------------------------------------------
# Calculate Return Rate
#----------------------------------------------------------------------------------------
data['RETURN RATE'] = (data['ITEMS RETURN COUNT'] / data['ITEM SOLD OUT COUNT']) * 100
data

df=data.copy()
# Count the number of fraudulent returns for each CUSTOMER, considering the count of items returned
fraudulent_returns_count = df[df['LABEL'] == 'Fraudulent'].groupby('CUSTOMER ID')['ITEMS RETURN COUNT'].sum().reset_index()
fraudulent_returns_count.columns = ['CUSTOMER ID', 'total_fraudulent_items_returned']

# Sum of total number of returns for each customer
total_returns_count = df.groupby('CUSTOMER ID')['ITEMS RETURN COUNT'].sum().reset_index()
total_returns_count.columns = ['CUSTOMER ID', 'total_number_of_all_returns']  

# Merge the fraudulent and total returns counts with the original DataFrame
df = pd.merge(df, fraudulent_returns_count, on='CUSTOMER ID', how='left')
df = pd.merge(df, total_returns_count, on='CUSTOMER ID', how='left')

# Fill NaN values with 0 (CUSTOMERs without fraudulent returns)
df['total_fraudulent_items_returned'] = df['total_fraudulent_items_returned'].fillna(0)

# Calculate Fraud Rate
df['Fraud Rate'] = (df['total_fraudulent_items_returned'] / df['total_number_of_all_returns']) * 100

# Setting Threshold for customer classification
df['customer_classification'] = np.where(df['Fraud Rate'] == 0, 'No Warning',
                                         np.where((df['Fraud Rate'] > 0) & (df['Fraud Rate'] <= 15), 'Warning',
                                                  np.where((df['Fraud Rate'] > 15) & (df['Fraud Rate'] <= 40), 'Ban for Couple of Years',
                                                           'Ban Completely')))


df.info()

# NULL values Check
df.isna().sum()

# Deal with Null Values Imputation so we data from loss
for col in df.columns:
    df[col]=df[col].interpolate()

df.isna().sum()
df.dropna(inplace=True)
df.isna().sum()

# Label Encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Items Type cat']= label_encoder.fit_transform(df['ITEM TYPE'])
df['Items Type cat'].unique()

df['Return Reason cat']= label_encoder.fit_transform(df['RETURN REASON'])
df['Return Reason cat'].unique()

df['Pre Verification cat']= label_encoder.fit_transform(df['PRE VERIFICATION'])
df['Pre Verification cat'].unique()

df['Return Request cat']= label_encoder.fit_transform(df['RETURN RATE'])
df['Return Request cat'].unique()

df['Post Verification cat']= label_encoder.fit_transform(df['LABEL'])
df['Post Verification cat'].unique()

df['customer_classification cat']=label_encoder.fit_transform(df['customer_classification'])
df['customer_classification cat'].unique()

df.isna().sum()
df.info()

X = df.select_dtypes(include=['float64', 'int64','int32'])
X=X.drop(columns=['RANDOM ', 'customer_classification cat'])
y=df['customer_classification cat']

#-------------------------------------------------------------------------------------
# Splitting the dataset
#------------------------------------------------------------------------------------

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

#---------------------------------------------------------------------------------
# select the best featutre among all
#----------------------------------------------------------------------------------
learner=ClassificationAlgorithms()
max_feature=12
selected_features,ordered_features,ordered_scores=learner.forward_selection(max_feature,X_train,y_train)


X=df[selected_features]
X

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)


#-------------------------------------------------------------------------------------
# Train and  Test Different Models
#-------------------------------------------------------------------------------------
d1="./Deliverable_3_output/"
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

dic = {
    DecisionTreeClassifier(criterion='entropy'): "Decision Tree",
    RandomForestClassifier(): "Random Forest",
    GaussianNB(): "Naive Bayes",
    MLPClassifier(): "Neural Network",
    SVC(): "Support Vector Machine"
}

# Loop over the classifiers and evaluate each one
for key, value in dic.items():
    key.fit(X_train, y_train)
    y_pred = key.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    f1 = f1_score(y_test, y_pred,average='weighted')
    
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {value}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f"{d1}_{value}_cm.png")
    plt.show()
    
    print(f"{value}\n")
    print(f"Accuracy  : {acc}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1-score  : {f1}")
    print("\n")
<<<<<<< HEAD

#----------------------------------------------------------------------------------------------
# Analysing Customer Complaints 
#----------------------------------------------------------------------------------------------
d2='./Customer_Complaint_Monitor/'
=======
  
#------------------------------------------
# Analysing Customer Complaints 
#-------------------------------------------
>>>>>>> 97269c479461a8a3ac2fcb6b0c8fa5b588bcb9a3
df['customer_complaints'] = pd.cut(df['RETURN RATE'], 
                                   bins=[-1, 20, 50, 80, 100, float('inf')],
                                   labels=['Much more satisfied', 'Getting wrong color and sizes', 'Wrong items delivered', 'Always gets wrong item and poor quality products as well', 'Extremely poor quality products'],
                                   right=False)

df['CUSTOMER COMPLAINT'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='CUSTOMER COMPLAINT', y='RETURN RATE', palette='Set1')
plt.title('Return Rate by Customer Complaints')
plt.xlabel('Customer Complaints')
plt.ylabel('Return Rate')
plt.xticks(rotation=45)
plt.grid(axis='y')
<<<<<<< HEAD
plt.savefig("{d2}_Return Rate by Customer Complaints.png")
plt.show()
=======
plt.savefig("Return Rate by Customer Complaints.png")
plt.show()
>>>>>>> 97269c479461a8a3ac2fcb6b0c8fa5b588bcb9a3
