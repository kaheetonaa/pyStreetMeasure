import numpy as np
import pyvista as pv

data=np.load('da3-output/exports/npz/results.npz')
ext=data['extrinsics'][0]
depth=data['depth'][0]
rgb=data['image'][0]
intr=data['intrinsics'][0]
conf=data['conf'][0]
q=ext[0:,:3]
t=ext[0:,3]
world=np.array([[[0,0,0,0,0,0,0] for j in range(len(depth[0]))] for i in range(len(depth))]).astype(float) #x,y,z,r,g,b,confidence

for v in range(len(depth)):
    for u in range(len(depth[0])):
        temp_0=depth[v][u]*np.linalg.inv(intr)
        temp_1=np.matmul(temp_0,np.transpose(np.atleast_2d(np.array([u,v,1]))))
        temp_2=np.matmul(q,temp_1)#rotate
        result=temp_2+np.transpose(np.atleast_2d(t))#translate
        for i in range(3):
            world[v][u][i]=result[i][0]
            world[v][u][i+3]=rgb[v][u][i]
        world[v][u][6]=conf[v][u]
world=world.reshape(len(depth)*len(depth[0]),7) #from [u,v,7] to [u*v,7]
pc=pv.PointSet(world[0:,:3]) # take x y z
pc.point_data['confidence']=world[0:,6]
pc.point_data['color']=255-world[0:,3:6]
plotter = pv.Plotter(shape=(1,2))
plotter.subplot(0,0)
plotter.add_points(pc,scalars='color',rgb=True)
plotter.subplot(0,1)
plotter.add_points(pc,scalars='confidence')
plotter.link_views()
plotter.show()
