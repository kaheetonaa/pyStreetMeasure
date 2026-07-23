1. download Mapillary data using Mapillary2QGIS plugin
2. from id list download original image from mapillary
3. use on-site measurement (GNSS and laser distance measure) "gt_sp6_piacenza"
4. find closest Mapillary point to ground truth point (closest 3 points, max distance 25m)
5. apply segmentation on selected mapillary images ()
6. get the road masks from segmentation and resize to DA360 input requirement ()
7. apply DA360 on selected images with the mask
8. using CloudCompy (python wrapper of CloudCompare) to measure the width of the road.
9. recover the scale from observations using linear regression with least squared optimization (x=model_width, y=actual width - y = x * scale + bias)

