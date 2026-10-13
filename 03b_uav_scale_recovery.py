# (s,t)= argmin s,t SUM(s*d_pred + t - d_ref)**2

import numpy as np
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch_uncertainty import TUTrainer
from torch_uncertainty.routines import RegressionRoutine
from torch_uncertainty.metrics.regression import MeanSquaredLogError
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import cv2

pred=np.load('LC3D/pred.npy')
true=np.load('LC3D/height-ref.npy')
pred=cv2.resize(pred,(8192,5460),interpolation=cv2.INTER_CUBIC)
_pred,_true=[],[]

#true would have shape [x,y,depth]
for i in range(true.shape[0]):
    _x,_y=int(true[i][0]),int(true[i][1])
    #_pred += [pred[_y+_j, _x+_k] for _j in range(-2, 3) for _k in range(-2, 3)]
    _pred += [pred[_y, _x]]
    _true+=[true[i][2]]
    
_pred,_true=torch.tensor(_pred).reshape(-1,1),torch.tensor(_true)
print(_pred)
pred_train,pred_test,true_train,true_test=train_test_split(
    _pred, _true, test_size=0.5, random_state=42)

pred_val,pred_test,true_val,true_test=train_test_split(
pred_test,true_test,test_size=0.5,random_state=42
)
print(pred_train.shape,true_train.shape)

train_ds = TensorDataset(pred_train, true_train)
val_ds = TensorDataset(pred_val, true_val)
test_ds = TensorDataset(pred_test, true_test)


model = nn.Linear(in_features=1, out_features=1)

def optim_regression(
    model: nn.Module,
):
    return optim.Adam(model.parameters(), lr=1e-2) 

mse_rel_metric = MeanSquaredLogError()

routine = RegressionRoutine(
    output_dim=1,
    model=model,
    loss=mse_rel_metric,
    optim_recipe=optim_regression(model),
    dist_family=None,
)

#criterion = nn.MSELoss()  # Mean Squared Error Loss
#optimizer = optim.SGD(model.parameters(), lr=1e-5)  # Stochastic Gradient Descent
datamodule = LightningDataModule.from_datasets(
    train_ds, val_dataset=val_ds, test_dataset=test_ds, batch_size=8, num_workers=0
)
epochs = 2000
trainer = TUTrainer(accelerator="cpu", devices=1, max_epochs=epochs, enable_progress_bar=True)
trainer.fit(model=routine,datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)

rescale=model(torch.from_numpy(pred).reshape(-1,1))
rescale=rescale.reshape(pred.shape[0],pred.shape[1])
rescale=rescale.detach().numpy()
np.save('LC3D/rescale.npy',rescale)
