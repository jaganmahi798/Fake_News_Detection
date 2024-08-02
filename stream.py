import streamlit as st
import pandas as pd
import seaborn as sns
import warnings
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack,csr_matrix
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier



train_df = pd.read_csv('train.csv', header=None)
valid_df = pd.read_csv('valid.csv', header=None)
test_df = pd.read_csv('test.csv', header=None)

train_df.columns=['ID','Label','Statement','Subject','Speaker','Speaker Job Title','State Info','Party Affiliation','Barely True','False','Half true','Mostly True','Pants on Fire','Context']
valid_df.columns=['ID','Label','Statement','Subject','Speaker','Speaker Job Title','State Info','Party Affiliation','Barely True','False','Half true','Mostly True','Pants on Fire','Context']
test_df.columns=['ID','Label','Statement','Subject','Speaker','Speaker Job Title','State Info','Party Affiliation','Barely True','False','Half true','Mostly True','Pants on Fire','Context']
# Assuming df is your DataFrame
st.title("FakeNews Detection")
nav=st.sidebar.radio("Navigation",["Home","Feature Importance List","Prediction","User Input"])

if nav == "Home":
    st.image("fakenews.webp",width=600)
    if(st.checkbox("show data")):
        st.table(train_df.head(50))
    if(st.checkbox("graph")):
        warnings.filterwarnings("ignore", category=UserWarning, module='streamlit')
        sns.countplot(x='Label',data=train_df,palette='hls')
        # fig, ax = plt.subplots()
        # ax.scatter([1, 2, 3], [1, 2, 3])
# other plotting actions...
        st.pyplot()
        # st.pyplot()

l1=['Subject','Speaker','Speaker Job Title','State Info','Party Affiliation','Context']
l2=['Barely True','False','Half true','Mostly True','Pants on Fire']
for column in l1:  # Columns 4-14
    train_df[column] = train_df[column].fillna('unknown')  # Replace null values with 'unknown'
    valid_df[column] = valid_df[column].fillna('unknown')  # Do the same for the validation set
    test_df[column] = test_df[column].fillna('unknown')  # And the test set
for column in l2:  # Columns 4-14
    mode_value = train_df[column].mode()[0]  # Get the mode of the column
    train_df[column] = train_df[column].fillna(mode_value)  # Replace null values with the mode
    valid_df[column] = valid_df[column].fillna(mode_value)  # Do the same for the validation set
    test_df[column] = test_df[column].fillna(mode_value)  # And the test set

# Define the preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(tokens)
 
def process():
    lst=['Statement','Subject','Speaker','Speaker Job Title','State Info','Party Affiliation','Context']
    for column in lst:
        train_df[column] = train_df[column].apply(preprocess_text)
        valid_df[column] = valid_df[column].apply(preprocess_text)
        test_df[column] = test_df[column].apply(preprocess_text)

    label_mapping = {
    'FALSE': 0,
    'half-true': 0,
    'barely-true': 0,
    'mostly-true': 1,
    'pants-fire': 0,
    'TRUE': 1
    }
# Convert labels to binary values
    train_df['Label'] = train_df['Label'].map(label_mapping)
    valid_df['Label'] = valid_df['Label'].map(label_mapping)
    test_df['Label'] = test_df['Label'].map(label_mapping)

# Vectorization


    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df[lst].apply(lambda x: ' '.join(x.astype(str)), axis=1))
    X_valid = vectorizer.transform(valid_df[lst].apply(lambda x: ' '.join(x.astype(str)), axis=1))
    X_test = vectorizer.transform(test_df[lst].apply(lambda x: ' '.join(x.astype(str)), axis=1))


    



# Assuming y_train, y_valid and y_test are your target variables for train, validation and test sets respectively
    

# Create an XGBoost model
    model = xgb.XGBClassifier()

# Fit the model
    model.fit(X_train, y_train)

# Get feature importances
    importances = model.feature_importances_

# Get feature names
    feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame of features and their importances
    features_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance in descending order
    features_df = features_df.sort_values(by='Importance', ascending=False)

# print(features_df)


# Add the columns to your feature matrix
    X_train = hstack((X_train, train_df[['Barely True','False','Half true','Mostly True','Pants on Fire']].values))
    X_valid = hstack((X_valid, valid_df[['Barely True','False','Half true','Mostly True','Pants on Fire']].values))
    X_test = hstack((X_test, test_df[['Barely True','False','Half true','Mostly True','Pants on Fire']].values))

    return features_df,X_train,X_valid,X_test



label_mapping = {
    'FALSE': 0,
    'half-true': 0,
    'barely-true': 0,
    'mostly-true': 1,
    'pants-fire': 0,
    'TRUE': 1
    }
# Convert labels to binary values
train_df['Label'] = train_df['Label'].map(label_mapping)
valid_df['Label'] = valid_df['Label'].map(label_mapping)
test_df['Label'] = test_df['Label'].map(label_mapping)


y_train = train_df['Label']
y_valid = valid_df['Label']
y_test = test_df['Label']



# Create an SVM classifier




if nav == "Feature Importance List":
    features_df,X_train,X_valid,X_test=process()
    st.table(features_df.head(10))
    top_features_df = features_df.head(10)  # Select the top 10 features
    plt.figure(figsize=(10, 8))
    plt.barh(top_features_df['Feature'], top_features_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances from XGBoost Model')
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    st.pyplot()

accuracies=[]  
if nav == "Prediction":
    select=st.selectbox("Classification ",["Logistic Regression","Random Forest","SVM","CART"])
    if(select=="Logistic Regression"):

        Lg=st.selectbox("Logistic Regression ",["Validation Accuracy","Testing Accuracy"])
        features_df,X_train,X_valid,X_test=process()
        log_reg = LogisticRegression(max_iter=10000)
# Fit the model
        log_reg.fit(X_train, y_train)
        if Lg=="Validation Accuracy":
            y_pred_valid_lg = log_reg.predict(X_valid)
            accuracy=accuracy_score(y_valid, y_pred_valid_lg)
            accuracies.append(accuracy)
            st.success(f"Logistic Regression Validation accuracy: {accuracy * 100}")
        if(Lg=="Testing Accuracy"):
            y_pred_test = log_reg.predict(X_test)
            st.success(f"Logistic Regression Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
    
    if(select=="Random Forest"):

        Rf=st.selectbox("Random Forest ",["Validation Accuracy","Testing Accuracy"])
        features_df,X_train,X_valid,X_test=process()
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        if(Rf=="Validation Accuracy"):
            y_pred_valid_rf= rf.predict(X_valid)
            accuracy=accuracy_score(y_valid, y_pred_valid_rf)
            accuracies.append(accuracy)
            st.success(f"Random Forest Validation accuracy: {accuracy * 100}")
        if(Rf=="Testing Accuracy"):
            y_pred_test =rf.predict(X_test)
            st.success(f"Random Forest Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
    if(select=="SVM"):
        svm1=st.selectbox("SVM",["Validation Accuracy","Testing Accuracy"])
        features_df,X_train,X_valid,X_test=process()
        svm = SVC()

# Fit the classifier to the training data
        svm.fit(X_train, y_train)
        if(svm1=="Validation Accuracy"):
            y_pred_valid_svm = svm.predict(X_valid)
            accuracy=accuracy_score(y_valid, y_pred_valid_svm)
            accuracies.append(accuracy)
            st.success(f"SVM Validation accuracy: {accuracy * 100}")
        if(svm1=="Testing Accuracy"):
            y_pred_test =svm.predict(X_test)
            st.success(f"SVM Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
    if(select=="CART"):

        cart1=st.selectbox("CART ",["Validation Accuracy","Testing Accuracy"])
        features_df,X_train,X_valid,X_test=process()
        cart = DecisionTreeClassifier(random_state=42)
        cart.fit(X_train, y_train)

        if(cart1=="Validation Accuracy"):
            y_pred_valid_cart = cart.predict(X_valid)
            accuracy=accuracy_score(y_valid, y_pred_valid_cart)
            accuracies.append(accuracy)
            st.success(f"Decision Tree Validation accuracy: {accuracy * 100}")
        if(cart1=="Testing Accuracy"):
            y_pred_test =cart.predict(X_test)
            st.success(f"Decision Tree Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")

        if (st.checkbox("Comparison Graph")):

           model_names = ['LR', 'RF', 'SVM', 'CART']

    # lr_acc=accuracy_score(y_test, y_pred_valid_lg )
    # rf_acc=accuracy_score(y_test, y_pred_valid_rf )
    # svm_acc=accuracy_score(y_test, y_pred_valid_svm)
    # cart_acc=accuracy_score(y_test, y_pred_valid_cart)
   
    # accuracies = [lr_acc,rf_acc,svm_acc,cart_acc]

           plt.bar(model_names, accuracies)
           plt.xlabel('Model')
           plt.ylabel('Accuracy')
           plt.title('Accuracy vs Model Graph(Validation)')
           st.pyplot()
    
            
if nav == "User Input":
    Statement=st.text_input("Enter your Statement")
    Subject = st.text_input("enter Subject ")
    Speaker = st.text_input("enter Spaeker")
    Job_title=st.text_input("Speaker's Job title")
    State_Info=st.text_input("State Info")
    Party_Affiliation=st.text_input("Party Affiliation")
    Barely_true=st.number_input("Barely True")
    False1=st.number_input("False")
    Half_true=st.number_input("half True")
    mostly_true=st.number_input("mostly True")
    pants_on_fire=st.number_input("pants_on_fire")
    Context=st.text_input("Context")
    

# Preprocess text
    
    if(st.checkbox("predict")):
        text_inputs = [
        Statement,
        Subject,
        Speaker,
        Job_title,
        State_Info,
        Party_Affiliation
        ]
        text = " ".join(text_inputs)
        text = preprocess_text(text)
        vectorizer = TfidfVectorizer()
        text_vectorized=vectorizer.fit_transform([text])
    
        numeric_inputs = np.array([
        Barely_true,
        False1,
        Half_true,
        mostly_true,
        pants_on_fire
        ]).reshape(1, -1)
        numeric_sparse = csr_matrix(numeric_inputs)
        combined_features = hstack([text_vectorized, numeric_sparse])
        features_df,X_train,X_valid,X_test=process()
        assert combined_features.shape[1] == X_train.shape[1]
        log_reg = LogisticRegression(max_iter=10000)
# Fit the model
        log_reg.fit(X_train, y_train)
        
        pred=log_reg.predict(combined_features)[0]
        st.success(f"News is {pred}")














        


