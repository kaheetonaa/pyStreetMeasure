from zensvi.download import MLYDownloader
from zensvi.cv import Segmenter
import pandas as pd
import numpy as np
import cv2
import open3d as o3d

segment=False

mly_api_key = "MLY|4463150933761310|5995ca3757fc4f9a9c8f5e96b2efaa03"  # Please register your own Mapillary API key
downloader = MLYDownloader(mly_api_key=mly_api_key)
#45.467670,9.179418?z=19
downloader.download_svi("zen_svi_out_mapillary", lat=45.467670,lon=9.179418,buffer=10)

if bool(segment)==True:
    segmenter = Segmenter(dataset="mapillary",
                          task="semantic" # or "panoptic"
                          )
    segmenter.segment("zen_svi_out_mapillary/mly_svi/batch_1", 
                      dir_image_output = "zen_svi_out_mapillary/segment/",
                      dir_summary_output = "zen_svi_out_mapillary/segment/summary/"
                      )

