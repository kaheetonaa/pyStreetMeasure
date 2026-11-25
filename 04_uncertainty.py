import uncertainties
import numpy as np
from uncertainties import unumpy,ufloat,umath
import math
### Pc=K[R@P+T]

"""  --lat 45.963455361111116 --lon 7.917383638888889 --omega -0.0001474591
126878891 --phi 0.07188948967356248 --kappa 0.11738322173593 --w 8192 --h 5460 --fx 5460 --fy 5460 --height_abs 2142.469 --dem LC3D/gem_export_ellipsoid_32.t
if --output LC3D/true0080.umath. --sample_size 15 --z_scale -1 --y_scale -1
"""

#omega=-0.0001474591 
#phi=-0.07188948967356248
#kappa=0.11738322173593 

def euler_to_rotation(omega, phi, kappa):
	#XYZ order
	#https://en.wikipedia.org/wiki/Rotation_matrix
	Rx=np.array([[1,0,0],[0,umath.cos(omega),-umath.sin(omega)],[0,umath.sin(omega),umath.cos(omega)]])
	Ry=np.array([[umath.cos(phi),0,umath.sin(phi)],[0,1,0],[-umath.sin(phi),0,umath.cos(phi)]])
	Rz=np.array([[umath.cos(kappa),-umath.sin(kappa),0],[umath.sin(kappa),umath.cos(kappa),0],[0,0,1]])
	return unumpy.matrix(Rz@Ry@Rx)
	
omega,phi,kappa=[12,30,18]

lat=45.963455361111116#in degrees
lon=7.917383638888889
height=2142.469

sigma_x=0.02792
sigma_y=0.02792
sigma_z=0.05152
sigma_angle=0.01 #±0,01°


# Define T R P
T= unumpy.umatrix([0,0,0],[sigma_x,sigma_y,sigma_z])#1x3
omega=umath.radians(ufloat(omega,sigma_angle))
phi=umath.radians(ufloat(phi,sigma_angle))
kappa=umath.radians(ufloat(kappa,sigma_angle))
print(euler_to_rotation(omega,phi,kappa))

