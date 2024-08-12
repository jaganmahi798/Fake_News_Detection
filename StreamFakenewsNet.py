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

import warnings





df=pd.read_csv("FakeNewsNet.csv")
df.rename(columns={'title': 'Statement'}, inplace=True)

st.title("FakeNews Detection")
nav=st.sidebar.radio("Navigation",["Home","Feature Importance List","Prediction","User Input"])

if nav == "Home":
    st.image("fakenews.webp",width=600)
    if(st.checkbox("show data")):
        st.table(df.head(50))
    if(st.checkbox("graph")):
        st.set_option('deprecation.showPyplotGlobalUse', False) 
        sns.countplot(x='real',data=df,palette='hls')
        st.pyplot()

df=df.dropna()



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
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(stemmed_tokens)

# Apply the preprocessing function to the text column



import re

def preprocess_url(url):
    # Remove http:// or https:// from the beginning of the URL
    url = re.sub(r'^https?:\/\/', '', url)
    
    # Remove www. from the beginning of the URL
    url = re.sub(r'^www\.', '', url)
    
    # Remove any trailing slashes
    url = re.sub(r'\/$', '', url)
    
    # Remove digits
    url = re.sub(r'\d', '', url)
    
    return url

def process():
    df['news_url'] = df['news_url'].apply(preprocess_url)
    df['Statement'] = df['Statement'].apply(preprocess_text)

    df['text'] = df['Statement'] + ' ' + df['news_url'] + ' ' + df['source_domain']

# Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

# Fit and transform the text data
    X_train = vectorizer.fit_transform(df['text'])

# Assign the target variable
    y_train = df['real']


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

    return vectorizer,X_train,features_df

y_train = df['real']


if nav == "Feature Importance List":
    _,X_train,features_df=process()
    st.table(features_df.head(10))
    top_features_df = features_df.head(10)  # Select the top 10 features
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 8))
    plt.barh(top_features_df['Feature'], top_features_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances from XGBoost Model')
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    st.pyplot()






if nav == "Prediction":
    select=st.selectbox("Classification ",["Logistic Regression","Random Forest","SVM","CART"])
    _,X_train,features_df=process()
    X_train1, X_temp1, y_train1, y_temp1 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

       # Further split the test data into 50% test and 50% validation (which makes it 10% of the original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp1, y_temp1, test_size=0.5, random_state=42)
    if(select=="Logistic Regression"):

        Lg=st.selectbox("Logistic Regression ",["Validation Accuracy","Testing Accuracy"])
        
        log_reg = LogisticRegression(max_iter=10000)
        # Fit the model
        log_reg.fit(X_train1, y_train1)
        if Lg=="Validation Accuracy":
            y_pred_valid_lg = log_reg.predict(X_val)
            accuracy=accuracy_score(y_val, y_pred_valid_lg)
            # accuracies.append(accuracy)
            st.success(f"Logistic Regression Validation accuracy: {accuracy * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            cm = confusion_matrix(y_val, y_pred_valid_lg)
            sns.heatmap(cm, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Validation Confusion Matrix for Logistic Regression')
            st.pyplot()
        if(Lg=="Testing Accuracy"):
            y_pred_test = log_reg.predict(X_test)
            st.success(f"Logistic Regression Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            cm = confusion_matrix(y_test, y_pred_test)
            sns.heatmap(cm, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Testing Confusion Matrix for Logistic Regression')
            st.pyplot()
        
    
    if(select=="Random Forest"):

        Rf=st.selectbox("Random Forest ",["Validation Accuracy","Testing Accuracy"])
        rf = RandomForestClassifier()
        rf.fit(X_train1, y_train1)

        if(Rf=="Validation Accuracy"):
            y_pred_valid_rf= rf.predict(X_val)
            accuracy=accuracy_score(y_val, y_pred_valid_rf)
            # accuracies.append(accuracy)
            st.success(f"Random Forest Validation accuracy: {accuracy * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            cm_rf = confusion_matrix(y_val, y_pred_valid_rf)
            sns.heatmap(cm_rf, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Validation Confusion Matrix for Random Forest')
            st.pyplot()
        if(Rf=="Testing Accuracy"):
            y_pred_test =rf.predict(X_test)
            st.success(f"Random Forest Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            cm_rf = confusion_matrix(y_test,  y_pred_test)
            sns.heatmap(cm_rf, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Testing Confusion Matrix for Random Forest')
            st.pyplot()
    if(select=="SVM"):
        svm1=st.selectbox("SVM",["Validation Accuracy","Testing Accuracy"])
        svm = SVC()

# Fit the classifier to the training data
        svm.fit(X_train1, y_train1)
        if(svm1=="Validation Accuracy"):
            y_pred_valid_svm = svm.predict(X_val)
            accuracy=accuracy_score(y_val, y_pred_valid_svm)
            # accuracies.append(accuracy)
            st.success(f"SVM Validation accuracy: {accuracy * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            cm_svm = confusion_matrix(y_val, y_pred_valid_svm)
            sns.heatmap(cm_svm, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Validation Confusion Matrix for Support Vector Machine')
            st.pyplot()
        if(svm1=="Testing Accuracy"):
            y_pred_test =svm.predict(X_test)
            st.success(f"SVM Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            cm_svm = confusion_matrix(y_test,  y_pred_test)
            sns.heatmap(cm_svm, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Testing Confusion Matrix for Support Vector Machine')
            st.pyplot()

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']


        accuracies1 = []


        for kernel in kernels:
             model = SVC(kernel=kernel)
             model.fit(X_train1, y_train1)
             y_pred = model.predict(X_test)
             accuracies1.append(accuracy_score(y_test, y_pred))


        df = pd.DataFrame({'Kernel': kernels, 'Accuracy': accuracies1})

        st.set_option('deprecation.showPyplotGlobalUse', False) 
        sns.barplot(x='Kernel', y='Accuracy', data=df)
        # st.success(kernels,accuracies1)
        st.pyplot()


    if(select=="CART"):

        cart1=st.selectbox("CART ",["Validation Accuracy","Testing Accuracy"])
        features_df,X_train=process()
        cart = DecisionTreeClassifier(random_state=42)
        cart.fit(X_train1, y_train1)

        if(cart1=="Validation Accuracy"):
            y_pred_valid_cart = cart.predict(X_val)
            accuracy=accuracy_score(y_val, y_pred_valid_cart)
            # accuracies.append(accuracy)
            st.success(f"Decision Tree Validation accuracy: {accuracy * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            cm_cart = confusion_matrix(y_val, y_pred_valid_cart)
            sns.heatmap(cm_cart, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Validation Confusion Matrix for CART')
            st.pyplot()

        if(cart1=="Testing Accuracy"):
            y_pred_test =cart.predict(X_test)
            st.success(f"Decision Tree Testing accuracy: {accuracy_score(y_test, y_pred_test) * 100}")
            st.set_option('deprecation.showPyplotGlobalUse', False) 
            cm_cart= confusion_matrix(y_test, y_pred_test)
            sns.heatmap(cm_cart, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Testing Confusion Matrix for CART')
            st.pyplot()


if nav == "User Input":
    Statement = st.text_input("Enter your Statement")
    Url =st.text_input("Enter url")
    Source_Domain = st.text_input("Enter Source Domain")
    Tweetnum=st.number_input("No of Tweets")

    if(st.checkbox("Predict")):
        Statement=preprocess_text(Statement)
        Url=preprocess_url(Url)
        text=[Statement+" "+Url+" "+Source_Domain]
        vectorizer,X_train, _ = process()
        text_transformed = vectorizer.transform(text)  # Transform the input text
        X_train1, X_temp1, y_train1, y_temp1 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        svm = SVC()

        # Fit the classifier to the training data
        svm.fit(X_train1, y_train1)
        pred = svm.predict(text_transformed)
        st.success(f"News is {'Real' if pred[0] else 'Fake'}")












