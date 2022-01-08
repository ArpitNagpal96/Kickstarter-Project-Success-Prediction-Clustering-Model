
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

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
    
# Load train and test dataset
df_train=pd.read_excel(r"C:\Users\91989\OneDrive\Desktop\McGill MMA\INSY662\Kickstarter.xlsx")

####################### Enter the test training set below #######################################
df_test=pd.read_excel("")

# Data pre-processing of test and train data
df_test=data_preprocessing_function(df_test)
df_train=data_preprocessing_function(df_train)

y_test= df_test['state']
y_train= df_train['state']
X_test= df_test[[cols for cols in df_test.columns if 'state' not in cols]]
X_train= df_train[[cols for cols in df_train.columns if 'state' not in cols]]

# Feature selection from the given dataset
selected_feature_list= feature_selection_function(X_train,y_train)
X_train = X_train[[cols for cols in X_train.columns if cols in selected_feature_list]]
X_test = X_test[[cols for cols in X_test.columns if cols in selected_feature_list]]

# Running the model using the optimized hyperparameters
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)
ann = MLPClassifier(alpha=0.0001, hidden_layer_sizes= 6, max_iter= 10000, solver= 'adam', learning_rate='adaptive').fit(scaled_X_train, y_train)
y_test_pred = ann.predict(scaled_X_test)
# Calculate the accuracy of the prediction
print(accuracy_score(y_test, y_test_pred))
print(f1_score(y_test,y_test_pred))


###################################### Clustering Model #######################################

# Using K-means clustering elbow method to determine optimal cluster value
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_train)
    Sum_of_squared_distances.append(km.inertia_)
labels =km.labels_
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Calculation of Silhouette score for the K-Means clustering
silhouette = silhouette_samples(df_train, labels)
silhouette_score(df_train, labels)

#Using numerical columns from the processed dataset for clustering
cols=['goal', 'static_usd_rate', 'create_to_launch_days',
       'launch_to_deadline_days']
numerical_df=df_train[cols]
scaler=StandardScaler()
df_kmeans=scaler.fit_transform(numerical_df)

# Formation of 2 PCA components from the numerical columns
pca = PCA(n_components=2)
df = pca.fit_transform(df_kmeans)
reduced_df = pd.DataFrame(df, columns=['PC1','PC2'])
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Using 4 cluster for K-means
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(reduced_df)
labels=model.predict(reduced_df)
reduced_df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)

# Output the number of datapoints for K-means clustering
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)
for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] )
plt.legend(u_labels)
plt.show()

df=df_train

df=df[cols]
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
standardized_x=pd.DataFrame(standardized_x,columns=df.columns)
df=standardized_x
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(df)
labels=model.predict(df)
df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Number of datapoints in cluster 1 (K Means):", count1)
print("Number of datapoints in cluster 2 (K Means):", count2)
print("Number of datapoints in cluster 3 (K Means):", count3)
print("Number of datapoints in cluster 4 (K Means):", count4)

# Parallel Combination graph for interpreting using the entire dataset
import plotly.express as px
from pandas.plotting import *
centroids = pd.DataFrame(kmeans.cluster_centers_)
px.parallel_coordinates(centroids,labels=df.columns,color=u_labels)