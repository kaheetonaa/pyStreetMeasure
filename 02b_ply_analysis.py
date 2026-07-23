import cloudComPy as cc
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from cloudComPy.minimalBoundingBox import findRotation
#------- hyperparameter-------
dist_close=10
dist_far=20
n_step=2
export=False
interval_step=(dist_far-dist_close)/n_step

def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList

def sorFilter(cloud):
    refCloud = cc.CloudSamplingTools.sorFilter(cloud)#sorFilter
    (noiseCloud, res) = cloud.partialClone(refCloud)
    return noiseCloud


def measure_width_from_ply(_id,cloud_file,dist_close,dist_far,n_step,export,interval_step):
    cloud = cc.loadPointCloud(cloud_file)
    cloud_np=cloud.toNpArrayCopy() #convert to numpy array
    cloud_np=cloud_np[cloud_np[:,2]>=dist_close]
    cloud_np=cloud_np[cloud_np[:,2]<dist_far]
    cloud_plane=cc.ccPointCloud()
    cloud_plane.coordsFromNPArray_copy(cloud_np)
    cloud_plane = sorFilter(cloud_plane)
    cloud_np=cloud_plane.toNpArrayCopy() #convert to numpy array
    
    #----camera height----------
    plane = cc.ccPlane.Fit(cloud_plane)
    _a,_b,_c,_d=plane.getEquation() #ax+by+cz=d
    cam_dist=math.sqrt(_d**2)/math.sqrt(_a**2+_b**2+_c**2)
    
    #-------visualize-----------
    if export==True:
        ret = cc.SavePointCloud(cloud_plane, "Piacenza/img/plane/"+str(_id)+".bin")
    #-------- statistic --------
    dist=[]
    for i in range(n_step):
        cloud_filtered=cloud_np[cloud_np[:,2]>=dist_close+i*interval_step]
        cloud_filtered=cloud_filtered[cloud_filtered[:,2]<dist_close+(i+1)*interval_step]
        cloud_filtered=np.array(removeOutliers(cloud_filtered[:,0],1.5))
        dist+=[cloud_filtered.max()-cloud_filtered.min()]
    dist=np.array(dist)/cam_dist #normalize distance by the height of the camera
    dist_mean=np.mean(dist)
    dist_std=np.std(dist)
    return [{"w_dist_"+str(j):dist[j] for j in range(len(dist))}|{"id":_id}]

df=pd.DataFrame(pd.read_csv("Piacenza/piacenza-mapillary-groundtruth-link.csv"))
list_id=list(df["id"])


remove=[414868547085447,462407905233694,424540882697668]

list_id = [x for x in list_id if x not in remove]
df=[]

for i in tqdm(range(len(list_id))):
    cloud_file="Piacenza/img/results/"+str(list_id[i])+"_pc_pred_DA360.ply"
    df+=measure_width_from_ply(list_id[i],cloud_file,dist_close,dist_far,n_step,export,interval_step)

df=pd.DataFrame(df)
print(df)
df.to_csv('Piacenza/width_from_MDE.csv')
print(df)
