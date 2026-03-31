# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Local Python
#     language: python
#     name: local-python
# ---

# %% jupyter={"source_hidden": false, "outputs_hidden": false}

def project_point_with_distortion(xyz_world, R, t, f, cx, cy, k):
    """Returns (u, v, z_cam) for a COLMAP 3D point."""
    X_cam = R @ xyz_world + t
    z = X_cam[2]
    if z <= 0:
        return None
    xn = X_cam[0] / z
    yn = X_cam[1] / z
    r2 = xn**2 + yn**2
    factor = 1 + k * r2
    u = f * xn * factor + cx
    v = f * yn * factor + cy
    return u, v, z
