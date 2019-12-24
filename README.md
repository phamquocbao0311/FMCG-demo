# FMCG-demo
This is a demo for FMCG using tkinter.
## Introduction
This project I will focus on dectecting object and classifying each object into distinct categories. Images could contain above 500 objects and each object look very similar or even identical, positioned in close proximity. I have to compare the rules which manufacture give with the arrangement of objects in image and make sure the image satisfies the rule.
## Getting started
First, you download the models and images follow the link: [here](https://drive.google.com/drive/folders/1HYADyV8-Hrd9Pvcq3XRVVh8-Mci8FD9X?usp=sharing) and put them into its directory.  
Then run the command: `python main.py`
## How does this demo work?
this demo have two step:  
- **Object detection:** I used retinanet to perform object detection. The model I trained on https://github.com/fizyr/keras-retinanet with 3K images. I splits 2K images for training and 1K images for testing. My model reach ~ 86% Map on test set.
- **Classification:** I used the state of the art ArcFace to classify object. Here the link I train the model: https://github.com/4uiiurz1/keras-arcface. Each object detected is put into ArcFace model and then I take 128 embedding vector to compare distance with label. 
## Status
using some pictures to show.

## Discussion

## Note
The data for training object detection is prohibited because of my company's policy.
