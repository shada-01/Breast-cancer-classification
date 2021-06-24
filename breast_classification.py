import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def init():

    cancer=load_breast_cancer()
    df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
    df1=pd.DataFrame(cancer.target,columns=['target'])
    df3=pd.concat([df,df1],axis=1)
    y=pd.Series(df3['target'])
    X=df3.drop('target',1)
    return X,y

def split_data(X,y):
        X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)
        return X_train, X_test, y_train, y_test

def knn_(X_train, X_test, y_train, y_test):
        knn=KNeighborsClassifier(n_neighbors=1)
        knn.fit( X_train, y_train)
       
        return knn



def accuracy_plot(X,y):
    
    X_train, X_test, y_train, y_test = split_data(X,y)

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = knn_(X_train, X_test, y_train, y_test)

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['skyblue','skyblue','lightgreen','lightgreen'])
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='black', fontsize=12)


    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    plt.show()


    
    
def main():
    X,y=init()
    X_train, X_test, y_train, y_test=split_data(X,y)
    knn=knn_(X_train, X_test, y_train, y_test)
    acc=knn.score(X_test,y_test)
    print(f"Score of KNN classifier = {acc}")
    accuracy_plot(X,y)

main()