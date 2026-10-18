"""Reproducible churn classification on INCLUDED SYNTHETIC customer records."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'synthetic_telco_customers.csv'
OUT=ROOT/'results'
SEED=42

def generate(n=7043):
    rng=np.random.default_rng(SEED)
    tenure=rng.integers(0,73,n)
    contract=rng.choice(['Month-to-month','One year','Two year'],n,p=[.55,.25,.20])
    internet=rng.choice(['Fiber optic','DSL','No'],n,p=[.47,.35,.18])
    payment=rng.choice(['Electronic check','Mailed check','Bank transfer','Credit card'],n)
    support=rng.choice(['Yes','No','No internet service'],n,p=[.40,.50,.10])
    monthly=np.round(np.clip(rng.normal(72,25,n)+8*(internet=='Fiber optic')-22*(internet=='No'),18,130),2)
    total=np.round(monthly*np.maximum(tenure,.35),2)
    senior=rng.binomial(1,.17,n)
    p_logit=(-1.1+1.2*(contract=='Month-to-month')-.65*(contract=='Two year')
        -0.037*tenure+0.55*(internet=='Fiber optic')-0.65*(support=='Yes')
        +0.25*(payment=='Electronic check')+0.29*senior+0.012*(monthly-70))
    p=1/(1+np.exp(-p_logit))
    churn=np.where(rng.random(n)<p,'Yes','No')
    return pd.DataFrame({'customerID':[f'SYN-{i:05d}' for i in range(n)],'gender':rng.choice(['Male','Female'],n),
        'SeniorCitizen':senior,'tenure':tenure,'Contract':contract,'InternetService':internet,
        'TechSupport':support,'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total,'Churn':churn})

def main():
    OUT.mkdir(exist_ok=True)
    if not DATA.exists():
        generate().to_csv(DATA,index=False)
    df=pd.read_csv(DATA)
    y=(df['Churn']=='Yes').astype(int)
    X=df.drop(columns=['Churn','customerID'])
    num=X.select_dtypes(include='number').columns.tolist()
    cat=X.select_dtypes(exclude='number').columns.tolist()
    prep=ColumnTransformer([
      ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
      ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    algorithms={
      'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),
      'Random Forest':RandomForestClassifier(n_estimators=180,min_samples_leaf=3,random_state=42,n_jobs=-1),
      'Neural Network (MLP)':MLPClassifier(hidden_layer_sizes=(32,16),max_iter=180,early_stopping=True,random_state=42)}
    rows=[]; predictions={}
    fig,ax=plt.subplots(figsize=(7,5))
    for name,algo in algorithms.items():
        pipe=Pipeline([('preprocessing',prep),('model',algo)])
        pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xte)
        proba=pipe.predict_proba(Xte)[:,1]
        rows.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),
        'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,proba)})
        predictions[name]=(pred,proba)
        RocCurveDisplay.from_predictions(yte,proba,name=name,ax=ax)
    ax.set_title('ROC curves — synthetic dataset');fig.tight_layout();fig.savefig(OUT/'roc_curves.png',dpi=160);plt.close(fig)
    scores=pd.DataFrame(rows).sort_values('roc_auc',ascending=False)
    scores.to_csv(OUT/'metrics.csv',index=False,float_format='%.6f')
    best=scores.iloc[0]['model']
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(yte,predictions[best][0],display_labels=['Stayed','Churned'],ax=ax,cmap='Blues')
    ax.set_title('Confusion matrix — '+best);fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));scores.set_index('model')[['accuracy','f1','roc_auc']].plot.bar(ax=ax,rot=12)
    ax.set_ylim(0,1);ax.set_ylabel('Score');ax.set_title('Measured model performance');fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=160);plt.close(fig)
    (OUT/'summary.txt').write_text('DATA: SYNTHETIC, NOT Kaggle/IBM ORIGINAL\n'+f'Rows: {len(df)} | Train: {len(Xtr)} | Test: {len(Xte)} | Seed: 42\n\n'+scores.to_string(index=False,float_format=lambda v:f'{v:.4f}')+'\n\nBest by ROC-AUC: '+best+'\n',encoding='utf-8')
    print((OUT/'summary.txt').read_text())

if __name__=='__main__': main()
