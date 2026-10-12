from zensvi.cv import Segmenter

segmenter = Segmenter(dataset="mapillary", # or "mapillary"
                      task="semantic" # or "panoptic"
                      )
<<<<<<< HEAD
<<<<<<< HEAD
segmenter.segment(dir_input="Piacenza/img/original/selected", 
                  dir_image_output = "Piacenza/img/mask",
                  dir_summary_output = "Piacenza/img/mask_summary"
                  )

=======
=======
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
segmenter.segment("Piacenza/img/original/selected", 
                  dir_image_output = "Piacenza/img/mask",
                  )
#------ to numpy with .... -------------------------
<<<<<<< HEAD
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
=======
>>>>>>> 78e47197e479dd1a2791c7c95229782b143f5fbf
