import uncertainties

### Pc=K[R@P+T]

"""  --lat 45.963455361111116 --lon 7.917383638888889 --omega -0.0001474591
126878891 --phi 0.07188948967356248 --kappa 0.11738322173593 --w 8192 --h 5460 --fx 5460 --fy 5460 --height_abs 2142.469 --dem LC3D/gem_export_ellipsoid_32.t
if --output LC3D/true0080.npy --sample_size 15 --z_scale -1 --y_scale -1
"""

omega=-0.0001474591 
phi=-0.07188948967356248
kappa=0.11738322173593 

sigma_x=0.02792
sigma_y=0.02792
sigma_z=0.05152
sigma_omega=0.01 #±0,01°
sigma_phi=0.01
sigma_kappa=0.01
