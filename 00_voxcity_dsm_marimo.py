import marimo

__generated_with = "0.23.6"
app = marimo.App()


app._unparsable_cell(
    r"""
    !earthengine authenticate --auth_mode=notebook
    4/1AXlqoi6yqTM2gaU8YKoi-5taidQzsA42PhcZ9FCLsoBc1F7ADQJoHXDGroY
    """,
    name="_"
)


@app.cell
def _():
    from voxcity.generator import get_voxcity
    import ee

    ee.Authenticate()

    a=ee.Initialize(project='kuquanghuy')
    print(a)
    return (get_voxcity,)


@app.cell
def _(get_voxcity):

    _lat=[45.90219,45.88690]#min,max
    _lng=[8.76709,8.79112]#min,max


    rectangle_vertices = [
        (_lng[0], _lat[0]), (_lng[0], _lat[1]),
        (_lng[1], _lat[1]), (_lng[1], _lat[0]),
    ]
    meshsize=30
    voxcity = get_voxcity(
        rectangle_vertices,
        meshsize,
        building_source = 'OpenStreetMap',
        land_cover_source='Dynamic World V1',
        canopy_height_source='ETH Global Sentinel-2 10m Canopy Height (2020)',
        dem_source='FABDEM',
    )

    print(voxcity)
    return


if __name__ == "__main__":
    app.run()
