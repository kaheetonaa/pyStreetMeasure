import pyvista
import numpy as np
pl=pyvista.Plotter()
pl.import_gltf('da3-output/scene.glb')
p0=[]
p1=[]
p_order=0
def callback(point):
    global p_order,p0,p1
    if p_order==0:
        p0=np.array(point)
    if p_order==1:
        p1=np.array(point)
        d=np.linalg.norm(p1-p0)
        print(d)
    p_order+=1
    if p_order>1:
        p_order=0


pl.enable_point_picking(callback=callback)
pl.show()
