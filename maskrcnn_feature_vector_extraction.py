# coding: utf-8

# # Dataset preparation and Mask R-CNN feature vector extraction

# Change path to Mask RCNN directory:

import os

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# import os
import sys
import json
import random
import math
import numpy as np
import skimage.io
import matplotlib

matplotlib.use('Agg')  # if you want to use it in docker without a display

import matplotlib.pyplot as plt

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
rooms = ["/home/workspace/data/GH30_office/", "/home/workspace/data/GH30_living/", "/home/workspace/data/GH30_kitchen/",
         "/home/workspace/data/KennyLab", "/home/workspace/data/Arena/"]


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# ## Run Object Detection

for dataset_path in rooms:

    scene_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    scene_paths = [s + '/rgb/' for s in scene_paths]

    for scene in scene_paths:
        # count images
        nr_images = 0
        if os.path.isdir(scene):
            print(scene)
            for i in os.listdir(scene):
                # check if file is a "png"
                if i.endswith(".png"):
                    nr_images = nr_images + 1

        print(nr_images)

    count = 0
    for scene in scene_paths:
        if os.path.isdir(scene):
            for i in os.listdir(scene):
                # check if file is a "png"
                try:
                    if i.endswith(".png"):
                        # file name without extension
                        file_id = i.split('.')[0]

                        # set paths
                        file_path = os.path.join(scene, i)

                        seq_path = os.path.join(scene, file_id + "_detections_ycbv")

                        json_path = os.path.join(seq_path, file_id + ".json")
                        label_path = os.path.join(seq_path, "labels.txt")
                        vis_path = os.path.join(seq_path, file_id + "_visualization.png")

                        if not os.path.exists(seq_path):
                            os.makedirs(seq_path)

                        # img = cv2.imread(file_path)
                        image = skimage.io.imread(file_path)
                        # plt.imshow(img)

                        # Run detection
                        results = model.detect([image], verbose=1)

                        # Visualize results
                        r = results[0]

                        pltret = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                             class_names, r['scores'], False)
                        pltret.savefig(vis_path, bbox_inches='tight')
                        pltret.close()

                        # save values to files
                        output_json_data = {"detections": []}

                        # prepare arrays of detection result values
                        bounding_boxes = []

                        for i in r['rois']:
                            bounding_boxes.append({'minX': int(i[1]),
                                                   'minY': int(i[0]),
                                                   'maxX': int(i[3]),
                                                   'maxY': int(i[2])})

                        labels = []

                        f = open(label_path, 'w')

                        for label_id in r['class_ids']:
                            label_name = class_names[label_id]
                            labels.append(label_name)

                            f.write(str(label_id) + ": " + label_name + "\n")

                        f.close()

                        scores = r['scores']

                        for d in range(len(r['scores'])):
                            output_json_data['detections'].append({'id': d,
                                                                   'bb': bounding_boxes[d],
                                                                   'label': labels[d],
                                                                   'score': str(scores[d]),
                                                                   'featureDimensions': [len(r['features'][d])]})

                            feature_path = os.path.join(seq_path, str(d) + ".feature")

                            temp_feature = []

                            # copy the values to a new list otherwise they are incomplete for some reason
                            for i in range(len(r['features'][d])):
                                temp_feature.append(r['features'][d][i])

                            # save one feature file for each detection
                            with open(feature_path, 'w') as f:
                                f.write(str(temp_feature))

                        with open(json_path, 'w') as output_json_file:
                            json.dump(output_json_data, output_json_file)

                        count = count + 1

                        print("Status: {0}".format(count / nr_images))

                except Exception as e:
                    print(e)
