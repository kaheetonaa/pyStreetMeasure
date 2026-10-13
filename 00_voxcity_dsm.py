from voxcity.generator import get_voxcity
from voxcity.exporter.obj import export_obj
import ee

ee.Authenticate()
ee.Initialize(project='ee-kuquanghuylcb')

_lat=[45.89338,45.90736]#min,max
_lng=[7.95181,7.98440]#min,max


rectangle_vertices = [
    (_lng[0], _lat[0]), (_lng[1], _lat[0]),
    (_lng[1], _lat[1]), (_lng[0], _lat[1]),
]
meshsize=10
model = get_voxcity(
    rectangle_vertices,
    meshsize,
    building_source = 'OpenStreetMap',
    land_cover_source='Dynamic World V1',
    canopy_height_source='ETH Global Sentinel-2 10m Canopy Height (2020)',
    dem_source='COPERNICUS',
)

print(model)
export_obj(model, output_dir='.', file_name='dsm')


