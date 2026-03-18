import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner="ramani0527", repo_name="MLOPS", mlflow=True)
mlflow.set_tracking_uri("https://github.com/ramani0527/mlops.git")
"""To rectify the error - The configured tracking 
uri scheme: 'file' is invalid for use with the proxy mlflow-artifact scheme. The allowed tracking schemes are: {'http', 'https'}
we are writing this line below"""

#Load wind dataset
wine= load_wine()
X= wine.data
y= wine.target
#train and test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state = 42)

# Define the params for ml model
max_d = 10
n_esti = 5
#Mention your experiment below or 
# a new experiment can be created automatically on this window by simply giving the above code line
mlflow.set_experiment("YT-MLOPS-exp-1")

# you can created another experiment and send it as parameter in the start_run  
# First run 
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_d, n_estimators=n_esti, random_state = 42)
    rf.fit(X_train,y_train)
    
    y_pred = rf.predict(X_test)
    accuracy= accuracy_score(y_pred,y_test)
    
    mlflow.log_param("max_depth",max_d)
    mlflow.log_param("n_estimators",n_esti)
    mlflow.log_metric('accuracy', accuracy)
    


    #Creating Confusion Matrix plot as second run 
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title('Confusion Matrix')
    #save.plot
    plt.savefig("Wine_Confusion_Matrix.png")
    
    #log actifacts using mlflow
    mlflow.log_artifact("Wine_Confusion_Matrix.png") 
    mlflow.log_artifact(__file__)
    
    #tags adding
    mlflow.set_tags({"Brand":"GM","year":2019})
    
    #log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-model")
    
    print('accuracy',accuracy)