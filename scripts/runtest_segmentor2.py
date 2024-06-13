# Development scripts for segmentor2

import numpy as np
import tifffile
import napari
import leopardgecko.segmentor2 as lgs2

import logging
logging.basicConfig(level=logging.INFO)

# # Load data and respective labels
# 
# data and labels filenames as a tuple. This is not required but it makes it easier to setup training with multiple volumes

data_labels_fn=[
    ("./scripts/test_data/TS_0005_crop.tif", "./scripts/test_data/TS_0005_ribos_membr_crop.tif"),
]

traindatas=[]
trainlabels=[]

for datafn0, labelfn0 in data_labels_fn:
     #Make sure data and labels are curated in the correct data format
    traindatas.append(tifffile.imread(datafn0))
    trainlabels.append(tifffile.imread(labelfn0)) #In this case labels are already in uint8

print(trainlabels[0].max())


segm2 = lgs2.cMultiAxisRotationsSegmentor2.create_simple_separate_models_per_axis(3)

segm2.NN1_train(traindatas, trainlabels)
