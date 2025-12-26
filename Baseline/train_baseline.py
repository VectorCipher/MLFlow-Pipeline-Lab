import matplotlib.pyplot as plt 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pathlib import Path
import pandas as pd
import numpy as np

## Path
ROOT = Path(__file__).parent
ARTIFACTS = ROOT/"artifacts"
ARTIFACTS.mkdir(exist_ok=True)

## ML Flow setup
mlflow.set_experiment("Baseline_model")

## Load the dataset
df = pd.read_csv("C:/Users/Naitik/OneDrive/Desktop/MLFlow-Pipeline-Lab/Data/smartphone_returns.csv")

## split the dataset

X = df.drop(columns=["returned"])
y = df["returned"].astype(int)

numeric_columns = X.select_dtypes(include=[np.number]).columns.to_list()
categorical_columns = X.select_dtypes(exclude=[np.number]).columns.to_list()

## Preprocessing

## Column Transformer apply different preprocessing steps to different columns of the same dataset
## Numerical columns -> Scaling 
## Categorical Columns -> One hot encoding 
## transformers [name,transformation_strategy,column_names]

preprocess = ColumnTransformer(
    transformers=[
        ("numerical_column",StandardScaler(with_mean=False),numeric_columns),
        ("categorical_column",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical_columns)
        
    ]
)

## Model and hyperparameter
N_ESTIMATORS = 200
MAX_DEPTH = 12
RANDOM_STATE = 333

## Declare the model 
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth= MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
)

## Declare the pipeline for the model using pipeline 
classification_model = Pipeline(
    [
        ("prep",preprocess),
        ("randomf",model)
    ]
)

## Train test split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=RANDOM_STATE)

## Run the MLFlow by run name 
with mlflow.start_run(run_name="baseline_randomf"):
    mlflow.log_param("n_estimators",N_ESTIMATORS)
    mlflow.log_param("max_depth",MAX_DEPTH)
    
    ##Fit
    classification_model.fit(X_train,y_train)
    
    ## Evaluation
    y_pred = classification_model.predict(X_test)
    y_prob = None
    
    ## To claculate about the AUC AND ROC
    try:
        y_prob = classification_model.predict_proba(X_test)[:,1]
        ## It provide confidence over the class 1 . 
    except Exception:
        pass
    
    ## It provide the Accuracy score and f1 score metric 
    acc_score = accuracy_score(y_test,y_pred)
    f1score = f1_score(y_test,y_pred)
    
    ## LOG THE METRICS 
    mlflow.log_metric("Accuracy",float(acc_score))
    mlflow.log_metric("F1",float(f1score))
    if y_prob is not None:
        auc = roc_auc_score(y_test,y_prob)
        mlflow.log_metric("auc",float(auc))


## ROC -> Receiver Operating Characterstics
## AUC -> Area Under Curve 

        from sklearn.metrics import RocCurveDisplay## Help in creating Curve display of ROC  & Avods manual Calculation Of FPR AND TPR
        fig_roc,ax_roc = plt.subplots(figsize=(5,4)) ## IT Provide us the figure size and plotting area inches (sizes)
        RocCurveDisplay.from_predictions(y_test,y_prob,ax=ax_roc) ## Take y_test as true labels and y_prob probab of positive class 
        roc_path = ARTIFACTS / "roc_curve.png" ## Save the roc image in the Artifacts 
        fig_roc.tight_layout()
        fig_roc.savefig(roc_path, dpi= 160)## means high resolution
        plt.close(fig_roc)
        mlflow.log_artifact(str(roc_path)) ## copy the artifact into the MLFlow run directory 
    
## As y is the target variable convert it into fraction rename the axis to class and index to pct
    class_balance = y.value_counts(normalize=True).rename_axis("class").reset_index(name="pct")
    class_balance_path = ARTIFACTS / "class_balance.csv" ## store the data into the ARTIFACTS 
    class_balance.to_csv(class_balance_path, index=False)## Change it to the csv file 
    mlflow.log_artifact(str(class_balance_path))## KLog the artifact into the MLFLOW run directory

## Confusion matrix in the case of classification model
    cm = confusion_matrix(y_test, y_pred) ## Calculate between y_test as true labels and y_pred 
    fig_cm, ax_cm = plt.subplots(figsize = (4,4))## Define the size of the figure and axis 
    im = ax_cm.imshow(cm, cmap = "Blues") ## Show c map 
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
## This loop will iterate through each row_index and column_index and cell value 
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j,i, int(v), ha="center", va = "center")
    fig_cm.colorbar(im, ax= ax_cm, fraction=0.046, pad=0.04)
    cm_path = ARTIFACTS / "confusion_matrix.png"## Save the file in the ARTIFACTS FILE 
    fig_cm.tight_layout()
    fig_cm.savefig(cm_path, dpi=160)
    plt.close(fig_cm)
    mlflow.log_artifact(str(cm_path))## Log the ARTIFACTS INTO THE MLFLOW run directory
## Examples needed for the MLFLOW UI as for which type of data is required only for documentation 
## A signature is a contract describing input schema and output schema 
## Log the model to the mlflow run directory 
    in_examples = X_train.head(5)
    signature = infer_signature(in_examples, classification_model.predict(in_examples))
    mlflow.sklearn.log_model(classification_model, name= "model", signature=signature,input_example=in_examples)

    print({"accuracy": acc_score, "f1": f1score})