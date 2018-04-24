"""
Mask R-CNN
Base Configurations class.
"""
import numpy as np



class Config(object):
    NAME = None
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    DETECTION_MIN_CONFIDENCE = 0.7

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001

    def __init__(self):
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT




    def display(self):
        """"Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")