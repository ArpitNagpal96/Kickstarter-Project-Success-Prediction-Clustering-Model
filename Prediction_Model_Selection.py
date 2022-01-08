
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from numpy import where
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

def data_preprocessing_function(df):
    
    # Unique values of project ID and project name
    df=df.drop(['project_id','name','deadline','state_changed_at','created_at','launched_at'], axis=1)

    # Removal of pledged column since it is not included at the project launch for which the prediction is to be made
    df=df.drop(['pledged', 'usd_pledged'], axis=1)

    # Removal of backers count column since the value is computed post the project is launched and funding is collected for the project
    df=df.drop(['backers_count'], axis=1)

    # Removal of columns related to state changed due to irrelevance related to project launch
    df=df.drop(['state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr',
           'state_changed_at_hr','launch_to_state_change_days'], axis=1)

    # Removal of unary value variable from the dataset
    df=df.drop(['disable_communication'], axis=1)
    
    # Drop the rows with state other than failed and successful
    removal_indexes=[]
    for i in range(df.shape[0]):
        if df['state'].iloc[i]!="failed" and df['state'].iloc[i]!="successful":
            removal_indexes.append(i)
    df=df.drop(removal_indexes)
    
    # Drop rows with NaN values within the dataset

    df=df.dropna()
    df.reset_index(drop=True)
    
    labelencoder=LabelEncoder()
    df['state']=labelencoder.fit_transform(df['state'])
    
    # Dummify remaining categorical variables (if any) present in the dataframe
    df_dummified=pd.get_dummies(df)
    
    df_dummified.corr()
    
    # Drop column spotlight since it is highly correlated with state column
    df_dummified=df_dummified.drop(['spotlight'], axis=1)
    
    # Removing other correlated columns from the dataset
    df_dummified=df_dummified.drop(['blurb_len', 'name_len'], axis=1)
    
    #Developing the isolation forest with given contamination value
    isolation_forest=IsolationForest(contamination=.1)
    pred=isolation_forest.fit_predict(df_dummified)
    score=isolation_forest.decision_function(df_dummified)
    anomaly_index=where(pred==-1)

    #remove the anomalies identified
    list_remove=list(df_dummified.iloc[anomaly_index].index)
    df_dummified.drop(list_remove, inplace=True)
    df_dummified.reset_index(drop=True)
    
    return(df_dummified)
    
def feature_selection_function(X_train, y_train):
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_train, y_train)
    selected_feat= X_train.columns[(sel.get_support())]
    print("Number of selected features: "+ str(len(selected_feat)))
    return(selected_feat)

df_train=pd.read_excel(r"C:\Users\91989\OneDrive\Desktop\McGill MMA\INSY662\Kickstarter.xlsx")

#Pre-processing the input dataset
df_train=data_preprocessing_function(df_train)

#Formation of target variable and predictors from the processed dataset
dep_var= df_train['state']
ind_var= df_train[[cols for cols in df_train.columns if 'state' not in cols]]

# Train and test split for model selection and analysis
X_train,X_test,y_train,y_test=train_test_split(ind_var,dep_var,test_size=0.3,random_state=0)

# Filtering predictors based on features selected
selected_feature_list= feature_selection_function(X_train,y_train)
X_train = X_train[[cols for cols in X_train.columns if cols in selected_feature_list]]
X_test = X_test[[cols for cols in X_test.columns if cols in selected_feature_list]]

accuracy_of_models=[]
f1_score_of_models=[]
list_of_models=["Gradient-Boosted Tree","Random Forest","Logistic Regression","ANN","K-NN"]

# Gradient Boosting Tree (GBT) Model
gbt = GradientBoostingClassifier()
model = gbt.fit(X_train, y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)
# Calculate the results of the prediction
accuracy_of_models.append(accuracy_score(y_test, y_test_pred))
f1_score_of_models.append(f1_score(y_test,y_test_pred))
print(accuracy_score(y_test, y_test_pred))
print(f1_score(y_test,y_test_pred))

# Random Forest Model
rforest = RandomForestClassifier()
model=rforest.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)
# Calculate the results of the prediction
print(accuracy_score(y_test, y_test_pred))
print(f1_score(y_test,y_test_pred))
accuracy_of_models.append(accuracy_score(y_test, y_test_pred))
f1_score_of_models.append(f1_score(y_test,y_test_pred))

# Logistic Regression Model
lr = LogisticRegression()
model = lr.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)
# Calculate the results of the prediction
print(accuracy_score(y_test, y_test_pred))
print(f1_score(y_test,y_test_pred))
accuracy_of_models.append(accuracy_score(y_test, y_test_pred))
f1_score_of_models.append(f1_score(y_test,y_test_pred))

# Artificial Neural Network (ANN) Model
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)
ann = MLPClassifier().fit(scaled_X_train, y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = ann.predict(scaled_X_test)
# Calculate the results of the prediction
print(accuracy_score(y_test, y_test_pred))
print(f1_score(y_test,y_test_pred))
accuracy_of_models.append(accuracy_score(y_test, y_test_pred))
f1_score_of_models.append(f1_score(y_test,y_test_pred))

# K-Nearest Neighbors (K-NN)
knn= KNeighborsClassifier()
model=knn.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(X_test)
# Calculate the results of the prediction
print(accuracy_score(y_test, y_test_pred))
print(f1_score(y_test,y_test_pred))
accuracy_of_models.append(accuracy_score(y_test, y_test_pred))
f1_score_of_models.append(f1_score(y_test,y_test_pred))

# Exporting the prediction results for the models
models_measurement=pd.DataFrame(list(zip(list_of_models,accuracy_of_models,f1_score_of_models)),columns=['Model','Accuracy','F1 Score'])
models_measurement.to_excel('Model_Comparision.xlsx',index=False)

# Hyperparameter Tuning for the ANN model
ind_var = ind_var[[cols for cols in ind_var.columns if cols in selected_feature_list]]
parameters = {"hidden_layer_sizes": list(range(1,22)),'max_iter':[10000],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive']}
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(ind_var)
ann = MLPClassifier()
grid = GridSearchCV(estimator=ann, param_grid=parameters , cv=3, verbose=True)
grid.fit(scaled_X_train,dep_var)
print(grid.best_params_)
print(grid.best_score_)