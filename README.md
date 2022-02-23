This code is based on the Matterport implementation of Mask-RCNN (https://github.com/matterport/Mask_RCNN).
We applied the following changes:
* We extended model.py to also return the second last feature layer (corresponds to the first fully connected layer of the Resnet101-FPN backbone) according to https://gist.github.com/kielnino/3965d905a921abbe29a7c89af5ba1114.
* Add a boolean argument to display_instances in visualize.py to disinguish if the generated plot should be returned or directly displayed. 

It inferes the detection results for all available png images and stores the following files in a folder:
* json-file: stores for each detected object an ID, the dimension of the extracted feature vector, the detection score, the bounding box dimension and the label
* png-file: shows the detected objects as overlay
* feature-file: one for each detected object exists (association via object ID). It contains the extracted feature vector from the second last layer of Mask R-CNN.

We created results using weights trained on COCO and YCBV (weights copied from Kiru Park, who trained them for the BOP challenge 2020 https://github.com/kirumang/Pix2Pose).
The models trained on COCO and YCBV can be found here: https://drive.google.com/drive/folders/1xuaapmgFZ7xaCqesYaOL1l0-9hv4NXCF?usp=sharing

Tested in docker environment using GPU:
```
docker build -t maskrcnn_feature_gpu .
nvidia-docker run -it --network=host -v /home/edith/Projects/maskrcnn_feature_vector_extraction/:/home/workspace/maskrcnn_feature -v /home/edith/liebnas_mnt/PlaneReconstructions/:/home/workspace/data --gpus 'all,"capabilities=compute,utility"' --rm  --name maskrcnn_feature_gpu maskrcnn_feature_gpu bash
```
In the docker container
```
python3 setup.py install
python3 maskrcnn_feature_vector_extraction_ycbv.py
```


This code was used to generate detection results for the baseline used in 

```
@article{patten2020dgcm,
  title={Where Does It Belong? Autonomous Object Mapping in Open-World Settings},
  author={Langer, Edith and Patten, Timothy and Vincze, Markus},
  journal={Frontiers in Robotics and AI},
  pages={???},
  year={2022},
  publisher={Frontiers}
}
```

Note: the values of the extracted feature vector may vary slightly depending on the used GPU



