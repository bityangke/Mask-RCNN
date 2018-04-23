import sys
import os
import numpy as np

from data import dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

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
class CocoDataset(dataset.Dataset):
    def load_coco(self, dataset_dir, subset, year="2017", class_ids=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        coco = COCO("{}/annotations/instance_{}{}.json".format(dataset_dir, subset, year))
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instacen_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shpae [height, width, instance_count] and
        # list of class IDs that correspond to each channel of the mask
        for annotation in annotations:
            class_id =