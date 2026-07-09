from zensvi.download import MLYDownloader
from zensvi.cv import DepthEstimator
from zensvi.transform import PointCloudProcessor
import pandas as pd
import numpy as np
import cv2
import open3d as o3d

mly_api_key = "MLY|4463150933761310|5995ca3757fc4f9a9c8f5e96b2efaa03"  # Please register your own Mapillary API key
downloader = MLYDownloader(mly_api_key=mly_api_key)
downloader.download_svi("zen_svi_out_mapillary", lat=45.6016237,lon=8.6367005,buffer=10)

