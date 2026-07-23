from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import math
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
groundtruth=pd.read_csv("Piacenza/piacenza-mapillary-groundtruth-link.csv")
w_mde=pd.read_csv("Piacenza/width_from_MDE.csv")
merge = pd.merge(groundtruth, w_mde, on='id', how='inner')
print(merge.columns)
result=[]
r2max=0
for i in tqdm(range(1000)):
    y=merge['mis_2']
    X=merge[['w_dist_'+str(j) for j in range(len(merge.columns)-6)]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    model = LinearRegression(fit_intercept=True,positive=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
    rmse = math.sqrt(mean_squared_error(y_true=y_test,y_pred=y_pred)) 
    r2=model.score(X_test,y_test)
    if r2>r2max:
        r2max=r2
        r2_optimized_model=model
    result+=[[r2,mae,rmse]]

result=np.array(result).mean(axis=0)
print("Average result(r2,mae,rmse):",result)
print("Intercept:", r2_optimized_model.intercept_)
print("Intercept:", r2_optimized_model.intercept_)
print("Coefficients:", r2_optimized_model.coef_)

