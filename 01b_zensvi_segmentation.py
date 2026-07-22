from zensvi.cv import Segmenter

segmenter = Segmenter(dataset="mapillary", # or "mapillary"
                      task="semantic" # or "panoptic"
                      )
segmenter.segment("Piacenza/img/original/test", 
                  dir_image_output = "Piacenza/img/mask",
                  dir_summary_output = "Piacenza/img/mask_summary"
                  )
