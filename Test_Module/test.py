from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pickle
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
import torchmetrics
from pytorch_tabular import TabularModel
from ff_utils import FeedForward, test_model, MyDataset



# Output: unique ID of the team
def getName():
    return "RACITI GABRIELE"

# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(df, clfName):
    if clfName.lower() == "lr":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/standardization.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        norm = pickle.load(open("./models/normalizer_l2.save", 'rb'))
        X = pd.DataFrame(norm.transform(X))
        dfNew = pd.concat([y, X], axis = 1)
        return dfNew
    elif clfName.lower() == "knr":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/minmax_scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        pca = pickle.load(open("./models/pca70minmax.save", 'rb'))
        X = pd.DataFrame(pca.transform(X))
        dfNew = pd.concat([y, X], axis = 1)
        return dfNew
    elif clfName.lower() =="svr":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/minmax_scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([y, X], axis = 1)
        return dfNew
    elif clfName.lower() =="rf":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/minmax_scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        pca = pickle.load(open("./models/pca30minmax.save", 'rb'))
        X = pd.DataFrame(pca.transform(X))
        dfNew = pd.concat([y, X], axis = 1)
        return dfNew
    elif clfName.lower() =="ff":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/minmax_scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([y, X], axis = 1)
        return dfNew
    elif clfName.lower() =="tb":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/standardization.save", 'rb'))
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns) 
        dfNew = pd.concat([y, X_scaled], axis=1)
        return dfNew
    elif clfName.lower() =="tf":
        X = df.drop(columns=['Year'])
        y = df['Year']
        scaler = pickle.load(open("./models/standardization.save", 'rb'))
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns) 
        dfNew = pd.concat([y, X_scaled], axis=1)
        return dfNew
    
    
          

# Input: Regressor name ("lr": Linear Regression, "SVR": Support Vector Regressor, ...)
# Output: Regressor object
def load(clfName):
    if (clfName.lower() == "lr"):
        clf = pickle.load(open("./models/linearRegression.save", 'rb'))
        return clf
    elif (clfName.lower() == "svr"):
        print("N.B.: E' preferibile inserire le predizioni riguardanti il modello SVR per ultime a causa della lentezza di predizione dovuta alla complessità del modello (circa 20 minuti per 25mila campioni sulla mia macchina). La scelta di non rendere il modello meno complesso per velocizzare le predizioni è stata fatta per massimizzare i risultati del modello. In scenari reali, si potrebbe preferire una via di mezzo o utilizzare modelli più veloci e performanti.")
        clf = pickle.load(open("./models/svr.save", 'rb'))
        return clf
    elif (clfName.lower() == "knr"):
        clf = pickle.load(open("./models/knn.save", 'rb'))
        return clf
    elif (clfName.lower() == "rf"):
        clf = pickle.load(open("./models/randomForest.save", 'rb'))
        return clf
    elif (clfName.lower() == "ff"):
        model = FeedForward(90, 128, 1)
        model.load_state_dict(torch.load('./models/ff.pth'))
        return model 
    elif (clfName.lower() == "tb"):
        model = TabularModel.load_model('./models/tabnet', map_location=torch.device('cpu'))
        return model
    elif (clfName.lower() == "tf"):
        model = TabularModel.load_model('./models/tabtransformer', map_location=torch.device('cpu'))
        return model
    else:
        return None
    
# Input: PreProcessed dataset, Regressor Name, Regressor Object 
# Output: Performance dictionary
def predict(df, clfName, clf):
    X = df.drop(columns=['Year'])
    y = df['Year']
    if clfName.lower() =="ff":
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        X = X.values
        y = y.values
        test_dataset = MyDataset(X,y)
        test_loader = DataLoader(test_dataset, batch_size=1)
        clf.to(device)
        labels, ypred = test_model(clf, test_loader, device)
        
        mse_metric = torchmetrics.MeanSquaredError().to(device)
        mae_metric = torchmetrics.MeanAbsoluteError().to(device)
        mape_metric = torchmetrics.MeanAbsolutePercentageError().to(device)
        r2_metric = torchmetrics.R2Score().to(device)

        mse = mse_metric(ypred, labels).item()
        mae = mae_metric(ypred, labels).item()
        mape = mape_metric(ypred, labels).item()
        r2 = r2_metric(ypred, labels).item()
    elif clfName.lower() =="tb":
        ypred = clf.predict(X)
        mse = mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)
        mape = mean_absolute_percentage_error(y, ypred)
        r2 = r2_score(y, ypred)
    elif clfName.lower() =="tf":
        ypred = clf.predict(X)
        mse = mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)
        mape = mean_absolute_percentage_error(y, ypred)
        r2 = r2_score(y, ypred)
    else:
        ypred = clf.predict(X)
        mse = mean_squared_error(y, ypred)
        mae = mean_absolute_error(y, ypred)
        mape = mean_absolute_percentage_error(y, ypred)
        r2 = r2_score(y, ypred)
    perf = {"mse": mse, "mae": mae, "mape": mape, "r2score": r2}
    return perf

