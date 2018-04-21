import sys
import os







############################################################
#  Configurations
############################################################
from mrcnn.config import Config
class Cococonfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    NAME = "coco"

    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 80

############################################################
#  Dataset
############################################################
class CocoDataset():
    pass