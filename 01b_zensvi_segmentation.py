from zensvi.cv import Segmenter

segmenter = Segmenter(dataset="mapillary", # or "mapillary"
                      task="semantic" # or "panoptic"
                      )
segmenter.segment("Piacenza/img/original/selected", 
                  dir_image_output = "Piacenza/img/mask",
                  )
#------ to numpy with .... -------------------------
