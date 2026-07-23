import cloudComPy as cc
import numpy as np
import pandas as pd
from tqdm import tqdm
#------- hyperparameter-------
dist_close=10
dist_far=20
n_step=3
export=False
interval_step=(dist_far-dist_close)/n_step

def measure_width_from_ply(cloud_file,dist_close,dist_far,n_step,export,interval_step):
    #--------read PLY------------
    cloud = cc.loadPointCloud(cloud_file)
    refCloud = cc.CloudSamplingTools.sorFilter(cloud)#sorFilter
    origCloud = refCloud.getAssociatedCloud()
    (noiseCloud, res) = origCloud.partialClone(refCloud)
    cloud_np=noiseCloud.toNpArrayCopy() #convert to numpy array

    #-------visualize-----------
    if export==True:
        cloud_np=cloud_np[cloud_np[:,2]>=dist_close]
        cloud_np=cloud_np[cloud_np[:,2]<dist_far]
        cloud.coordsFromNPArray_copy(cloud_np)
        ret = cc.SavePointCloud(cloud, "dataSample.bin")

    #-------- statistic --------
    dist=[]
    for i in range(n_step):
        cloud_filtered=cloud_np[cloud_np[:,2]>=dist_close+i*interval_step]
        cloud_filtered=cloud_filtered[cloud_filtered[:,2]<dist_close+(i+1)*interval_step]
        dist+=[cloud_filtered[:,0].max()-cloud_filtered[:,0].min()]
    dist=np.array(dist)
    dist_mean=np.mean(dist)
    dist_std=np.std(dist)
    return [{"w_sc_amb_mean":dist_mean,"w_sc_amb_std":dist_std}]

df=pd.DataFrame(pd.read_csv("Piacenza/piacenza-mapillary-groundtruth-link.csv"))
list_id=list(df["id"])

df=[]

for i in tqdm(range(len(list_id))):
    cloud_file="Piacenza/img/results/"+str(list_id[i])+"_pc_pred_DA360.ply"
    df+=measure_width_from_ply(cloud_file,dist_close,dist_far,n_step,export,interval_step)

df=pd.DataFrame(df)
print(df)
