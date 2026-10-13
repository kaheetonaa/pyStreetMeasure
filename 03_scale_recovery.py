# (s,t)= argmin s,t SUM(s*d_pred + t - d_ref)**2

import numpy as np
from sklearn.model_selection import train_test_split
from lightning import LightningDataModule
import torch
import torch.nn as nn
import torch.optim as optim
from torch_uncertainty import TUTrainer
from torch_uncertainty.routines import RegressionRoutine
from torch_uncertainty.metrics.regression import MeanSquaredErrorInverse
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import cv2

pred=np.load('UAVMapping/pred.npy')
true=np.load('UAVMapping/true.npy')
pred=cv2.resize(pred,(true.shape[1],true.shape[0]),interpolation=cv2.INTER_CUBIC)
_pred,_true=[],[]

for i in range(true.shape[0]):
	for j in range(true.shape[1]):
		if true[i,j]>0:
			_pred+=[pred[i,j]]
			_true+=[true[i,j]]	

_pred,_true=torch.tensor(_pred).reshape(-1,1),torch.tensor(_true)

pred_train,pred_test,true_train,true_test=train_test_split(
    _pred, _true, test_size=0.95, random_state=42)

pred_val,pred_test,true_val,true_test=train_test_split(
pred_test,true_test,test_size=0.8,random_state=42
)
print(pred_train.shape,true_train.shape)

train_ds = TensorDataset(pred_train, true_train)
val_ds = TensorDataset(pred_val, true_val)
test_ds = TensorDataset(pred_test, true_test)


model = nn.Linear(in_features=1, out_features=1)

def optim_regression(
    model: nn.Module,
):
    return optim.Adam(model.parameters(), lr=0.01) 

mse_rel_metric = MeanSquaredErrorInverse()

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
    train_ds, val_dataset=val_ds, test_dataset=test_ds, batch_size=64, num_workers=0
)
epochs = 100
trainer = TUTrainer(accelerator="cpu", devices=1, max_epochs=epochs, enable_progress_bar=True)
trainer.fit(model=routine,datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)
#for epoch in range(epochs):
#    # Forward Pass: Compute predicted y by passing x to the model
#    predictions = model(pred_train)
    
    # Compute the MSE loss
#    loss = criterion(predictions, true_train)
    
    # Backward Pass: Clear existing gradients, compute new gradients, update weights
#    optimizer.zero_grad()  # Reset gradients from the last step
#    loss.backward()        # Calculate gradients via backpropagation
#    optimizer.step()       # Update the weights
    
    # Track progress
#    if (epoch + 1) % 20 == 0:
#        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")




#for name, param in model.named_parameters():
#    print(name, param.data)

#mse=criterion(model(pred_test),true_test)
#print(mse)
rescale=model(torch.from_numpy(pred).reshape(-1,1))
rescale=rescale.reshape(pred.shape[0],pred.shape[1])
rescale=rescale.detach().numpy()
np.save('UAVMapping/rescale.npy',rescale)
plt.imshow(rescale-true)
plt.show()
