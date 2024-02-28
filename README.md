Breast Cancer Wisconsin (Diagnostic) Data Set
![image](https://github.com/sturmanidze/Cancer-Prediction-/assets/71189710/e14b45f7-7517-4f73-a097-6857a239e114)

The Breast Cancer dataset is available on the UCI machine learning repository managed by the University of California, Irvine. It comprises of 569 samples of malignant and benign tumor cells. The first two columns of the dataset contain the unique ID numbers of the samples and the corresponding diagnosis (M=malignant, B=benign), respectively. Columns 3-32 store 30 real-value features that have been calculated from digitized images of the cell nuclei. These features can be used to develop a model that can predict whether a tumor is benign or malignant. 

1= Malignant (Cancerous) - Present (M) 0= Benign (Not Cancerous) -Absent (B)

Attribute Information:

1. ID number 
2. Diagnosis (M = malignant, B = benign)
3-32. Ten real-valued features are calculated for each cell nucleus:
   a) radius (mean of distances from center to points on the perimeter)
   b) texture (standard deviation of gray-scale values) 
   c) perimeter 
   d) area 
   e) smoothness (local variation in radius lengths) 
   f) compactness (perimeter^2 / area - 1.0) 
   g) concavity (severity of concave portions of the contour) 
   h) concave points (number of concave portions of the contour) 
   i) symmetry 
   j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were calculated for each image, resulting in 30 features. For example, field 3 is Mean Radius, field 13 is Radius SE, and field 23 is Worst Radius. 

All feature values are recoded with four significant digits. The dataset doesn't have any missing attribute values. The class distribution is 357 benign and 212 malignant.

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from pandas.plotting import scatter_matrix
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set_style("dark")
import warnings
warnings.filterwarnings('ignore')
#Import datset
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head(10)

data.info()
data.drop(['Unnamed: 32','id'], axis = 1 , inplace=True)
data.describe()
data.skew()
Exploratory Data Analysis (EDA)
add Codeadd Markdown
#Visualizing Multidimensional Relationships
plt.style.use('fivethirtyeight')
sns.set_style("white")
sns.pairplot(data[[data.columns[0], data.columns[1],data.columns[2],data.columns[3],
                     data.columns[4], data.columns[5]]], hue = 'diagnosis' , size=3)
add Codeadd Markdown
#create the correlation matrix heat map
plt.figure(figsize=(10,6))
sns.heatmap(data[[data.columns[0], data.columns[1],data.columns[2],data.columns[3],
                     data.columns[4], data.columns[5]]].corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
plt.suptitle('Correlation Matrix')
​
add Codeadd Markdown
Data Preprocessing¶
add Codeadd Markdown
Transform the 'M' and 'B' values (target variable) to 1 and 0 respectively. Following the encoding of the categorical features, we will continue with the normalization (scalling) of the numerical features. For this we will use the MinMax scalling method.

add Codeadd Markdown
# Transform the 'yes' and 'no' values (target variable) to 1 and 0 respectively
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
​
#Scalling
scaler =MinMaxScaler(feature_range=(0, 1))
scaled_data =  pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
​
# Split the data to train and test sets
X = scaled_data.loc[:, scaled_data.columns != 'diagnosis']
y = scaled_data['diagnosis']
add Codeadd Markdown
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
add Codeadd Markdown
Model Evaluation Metrics¶
For model evaluation and To perform a full ROC analysis let's define two functions

add Codeadd Markdown
#Defining model evaluation function
def getModelEvaluationMetrics(classifier, model_name: str, x_test: pd.core.frame.DataFrame,
                              y_test: pd.core.frame.DataFrame, y_predicted, plot_confusion_matrix=False,
                              figsize=(10, 8)) -> np.ndarray:
​
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predicted)
    print('Confusion matrix:\n\n {0}'.format(conf_mat))
​
    if plot_confusion_matrix:
        labels = ['M', 'B']
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_mat, cmap=plt.cm.Reds)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.style.use('fivethirtyeight')
        sns.set_style("white")
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        plt.title(f'Confusion Matrix for {model_name}', fontweight='bold')
        plt.show()
​
    # Calculating the precision (tp/tp+fp)
    precision = str(np.round((conf_mat[1][1] / (conf_mat[1][1] +
                              conf_mat[0][1])) * 100, 2))
    print('The precision is: {0} %'.format(precision))
​
    # Calculating the recall (tp/tp+fn)
    recall = str(np.round((conf_mat[1][1] / (conf_mat[1][1] +
                           conf_mat[1][0])) * 100, 2))
    print('The recall is: {0} %'.format(recall))
​
    return conf_mat
add Codeadd Markdown
#Defining function for performing a full ROC analysis
def createROCAnalysis(classifier, model_name: str, y_test: pd.core.series.Series, pred_probs: np.ndarray,
                      plot_ROC_Curve=False, figsize=(10, 8)) -> int:
   
    if plot_ROC_Curve:
        plt.figure(figsize=figsize)
        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill Classifier')
        fp_rate, tp_rate, _ = roc_curve(y_test, pred_probs[:, 1])
        plt.plot(fp_rate, tp_rate, marker='.', label=model_name)
        plt.style.use('fivethirtyeight')
        sns.set_style("white")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}', fontweight='bold')
        plt.grid(True, alpha=0.1, color='black')
        plt.legend(loc='lower right')
        plt.show()
​
    # Calculate Area Under Curve (AUC) for the Receiver Operating
    # Characteristics Curve (ROC)
    auc_score = np.round(roc_auc_score(y_test, pred_probs[:, 1]), 4)
    print(f'{model_name} - ROC AUC score: {auc_score}')
​
    return auc_score
add Codeadd Markdown
Breast Cancer Prediction
add Codeadd Markdown
# Instantiate the Random Forest model
#Pre-tuned Hyperparameter of Random Forest Classifier on this dataset
​
rf_class = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=10, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=20,
            min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
# Assign the above probabilities to the corresponding class ('no', 'yes')
rf_class.fit(X_train, y_train)
rf_y_pred = rf_class.predict(X_test)
# Evaluate the model by using Recall/Precission:
getModelEvaluationMetrics(classifier=rf_class, model_name='Random Forest',x_test=X_test, y_test=y_test,
                              y_predicted=rf_y_pred, plot_confusion_matrix=True, figsize=(8,6))
add Codeadd Markdown
# Evaluate the model by using ROC Curve:
rf_pred_probs = rf_class.predict_proba(X_test)
createROCAnalysis(classifier=rf_class, model_name='Random Forest', y_test=y_test, pred_probs=rf_pred_probs,
                  plot_ROC_Curve=True, figsize=(8,6))
add Codeadd Markdown
Feature decomposition using Principal Component Analysis( PCA)
add Codeadd Markdown
The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.

add Codeadd Markdown
feature_importance = {}
best_estimator_fi = rf_class.feature_importances_
​
for feature, importance in zip(X_train.columns, best_estimator_fi):
    feature_importance[feature] = importance
​
importances = pd.DataFrame.from_dict(feature_importance, orient='index').rename(columns={0: 'Gini Score'})
​
importances = importances.sort_values(by='Gini Score', ascending=False)
# Plot for feature importance
plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')
sns.set_style("white")
sns.barplot(x=importances.index[0:10],
            y=importances['Gini Score'].iloc[0:10], palette='muted')
plt.title(f'Importance for the Top 10 Features (Gini criterion) ',
          fontweight='bold')
plt.grid(True, alpha=0.1, color='black')
plt.show()
add Codeadd Markdown
Let’s evaluate the same algorithms with a standardized copy of the dataset. Here, I use sklearn to scale and transform the data such that each attribute has a mean value of zero and a standard deviation of one.

add Codeadd Markdown
sc = StandardScaler()
X_s = sc.fit_transform(X)
​
pca = PCA(n_components = 10)
X_pca = pca.fit_transform(X_s)
​
PCA_df = pd.DataFrame()
​
PCA_df['PCA_1'] = X_pca[:,0]
PCA_df['PCA_2'] = X_pca[:,1]
​
plt.plot(PCA_df['PCA_1'][data.diagnosis == 1],PCA_df['PCA_2'][data.diagnosis == 1],'o', alpha = 0.7, color = 'r')
plt.plot(PCA_df['PCA_1'][data.diagnosis == 0],PCA_df['PCA_2'][data.diagnosis == 0],'o', alpha = 0.7, color = 'b')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(['Malignant','Benign'])
plt.show()
add Codeadd Markdown
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
​
pca = PCA(n_components = 10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Assign the above probabilities to the corresponding class ('no', 'yes')
rf_class.fit(X_train, y_train)
rf_y_pred = rf_class.predict(X_test)
# Evaluate the model by using Recall/Precission:
getModelEvaluationMetrics(classifier=rf_class, model_name='Random Forest',x_test=X_test, y_test=y_test,
                              y_predicted=rf_y_pred, plot_confusion_matrix=True, figsize=(8,6))
add Codeadd Markdown
# Evaluate the model by using ROC Curve:
rf_pred_probs = rf_class.predict_proba(X_test)
createROCAnalysis(classifier=rf_class, model_name='Random Forest', y_test=y_test, pred_probs=rf_pred_probs,
                  plot_ROC_Curve=True, figsize=(8,6))
​
add Codeadd Markdown
Conclution
*Using PCA I used only 10 components (important) from the dataset, though it contains 30 components! However, with PCA+RF the model slightly outweigh in recall, precision and ROC AUC score than that of the previous model. *

Random Forest Model Prediction without PCA
The precision is: 95.24 % The recall is: 93.02 % ROC AUC score: 0.9954

Random Forest Model Prediction with PCA
The precision is: 97.62 % The recall is: 95.35 % ROC AUC score: 0.9974
