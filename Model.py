import pandas as pd   # Data Preprocessing
import numpy as np    # Mathematical Computation
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle    # Save the model

# Read the datset
df = pd.read_csv('Model_df.csv')
# Top 5 rows
print(df.head())

# Column Names
print(df.columns)   # 

# Shape
print(df.shape)   

# Data Preprocessing
# 1) Handling Null Values
print(df.isnull().sum())      # There are no null values

# 2) Handle the duplicates
print(df.duplicated().sum())  # Total number of dupliactes



# 4) Check the data types
print(df.dtypes) 

# 5) Checking the target variable
print(df['Profit'].value_counts())

# Select x (independenat feature) and y (dependent feature)
x = df.drop(['Profit'],axis=1)
y = df['Profit']

print(type(x))    # Dataframe
print(type(y))    # Series
print(x.shape)     
print(y.shape)    


# Split the data into train and test data
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
print(x_train.shape)  
print(x_test.shape)       
print(y_train.shape)       
print(y_test.shape)   

# Decision Tree Classifier 
dtc = DecisionTreeClassifier(criterion='entropy',max_depth=12,min_samples_split=25)
# Train the classifier on the training data
dtc.fit(x_train,y_train)
# ML Model Building

lr = LinearRegression()
ada_boost = AdaBoostClassifier(base_estimator=dtc,n_estimators=52)
#knn = KNeighborsClassifier(n_neighbors=11)

lr.fit(x_train,y_train)
ada_boost.fit(x_train,y_train)


print('Test Score LR',lr.score(x_test,y_test))
print('Test Score AdaBoost',ada_boost.score(x_test,y_test))
#print('Test Score KNN',knn.score(x_test,y_test))

# Saving the  model
pickle.dump(lr,open('lr_model.pkl','wb'))
pickle.dump(ada_boost,open('ada_boost_model.pkl','wb'))