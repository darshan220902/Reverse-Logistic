import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv('return_rates.csv')
df

df.info()
#-------------------------------------------------------------------
# imputation of missing values
#-------------------------------------------------------------------

for col in df.columns:
    df[col]=df[col].interpolate()

df.info()
df.dropna(inplace=True)

df.info()
df.columns

df['CUSTOMER ID'].value_counts()




df['RETURN DATE']=pd.to_datetime(df['RETURN DATE'])
df['SELLER ID'].unique()
df['LABEL'].value_counts()


di="./Return_rate_Visualization/"
dii="./Deliverable_1_output/"
#---------------------------------------------------
# Visualize return rates by RETURN REASON
#---------------------------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='RETURN REASON', hue='LABEL')
plt.title('Return Rates by Return Reason')
plt.xlabel('Return Reason')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig(f"{di}Return Rates by Return Reason.png")
plt.show()

#-----------------------------------------------------------------
# Visualize return rates by RETURN CONDITION
#-----------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='RETURN CONDITION', hue='LABEL')
plt.title('Return Rates by Return Condition')
plt.xlabel('Return Condition')
plt.ylabel('Count')
plt.savefig(f"{di}Return Rates by Return Condition.png")
plt.show()

#-------------------------------------------------------------------
# Visualize return rates by RETURN TYPE
#--------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='RETURN TYPE', hue='LABEL')
plt.title('Return Rates by Return LABEL')
plt.xlabel('Return LABEL')
plt.ylabel('Count')
plt.savefig(f"{di}Return Rates by Return Label.png")
plt.show()
#--------------------------------------------------------------------
# LABEL ENCODING
#--------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['RETURN CONDITION CAT'] = label_encoder.fit_transform(df['RETURN CONDITION'])
df['RETURN REASON CAT'] = label_encoder.fit_transform(df['RETURN REASON'])
df['RETURN TYPE CAT'] = label_encoder.fit_transform(df['RETURN TYPE'])
df['LABEL CAT'] = label_encoder.fit_transform(df['LABEL'])
df

#-------------------------------------------------------------------
# Take Feature
#-------------------------------------------------------------------

X = df[['RETURN CONDITION CAT','RETURN REASON CAT', 'RETURN TYPE CAT']]
y=df['LABEL CAT']

#-------------------------------------------------------------------
# Train & Test Split
#-------------------------------------------------------------------

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)



#-------------------------------------------------------------------
# Train Various Model & Test them on various Metric
#-------------------------------------------------------------------

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
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {value}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(f"{dii}{value}_cm.png")
    plt.show()
    
    print(f"{value}\n")
    print(f"Accuracy  : {acc}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1-score  : {f1}")
    print("\n")

<<<<<<< HEAD
=======
    
Product=df['PRODUCT ID'].astype(str)
product_ids = Product
output_directory = "./output"  # Output directory for QR codes
qr_mapping = generate_qr_codes(product_ids, output_directory)

# Print the mapping of product IDs to file paths
for product_id, file_path in qr_mapping.items():
    print(f"Product ID: {product_id}, QR Code File Path: {file_path}")
>>>>>>> 97269c479461a8a3ac2fcb6b0c8fa5b588bcb9a3
