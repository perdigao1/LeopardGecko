'''
Copyright 2022 Rosalind Franklin Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


"""
TODO:
Segmentation smilar to segmentor.py but with differences, based in kaggle experience

- Remove dependency to volume segmantics. bring unet code to here
- different models for different axis (multiple models)
- Save also NN2 model (MLP classifier)
- NN2 to pytorch, no sklearn
- segmentor file extension lgsegm2
- Revise metrics, make them more efficient
    - use the per-class metrics, and average by counting voxel-wise TP,FP,TN, FN
- support napari types, dask, xarray ?


"""
import numpy as np
import dask.array as da
#import subprocess
import tempfile
from pathlib import Path
import os
cwd = os.getcwd()
import tempfile
import logging
from types import SimpleNamespace
import tqdm #progress bar in iterations
from . import utils

#from . import metrics
#from .utils import *

import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
import torch.nn as nn
import albumentations as alb
import albumentations.pytorch

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

class cSegmentor2_Settings:
    """
    Settings only. Avoid functions. This will be pickled for loading and saving
    """

    #data_im_dirname': 'data'
    #seg_im_out_dirname': 'seg',
    #'model_output_fn': 'trained_2d_model',
    #'clip_data': False,
    data_vol_norm_process = "mean_stdev_3" #standard clipping

    #'st_dev_factor': 2.575,
    #'data_hdf5_path': '/data',
    #'seg_hdf5_path': '/data',
    #'training_axes': 'All',
    # image_size: 256, #not sure what this does, resizes bigger to 256x256 and smaller to 256x256?
    #'downsample': False,
    #'training_set_proportion': 0.8,
    cuda_device=0

    nn1_num_cyc_frozen=8
    nn1_num_cyc_unfrozen=5
    #patience=3

    nn1_loss_criterion='DiceLoss'
    #'alpha': 0.75,
    #'beta': 0.25,

    nn1_eval_metric='MeanIoU'
    #'pct_lr_inc': 0.3,
    #'starting_lr': '1e-6',
    #'end_lr': 50,
    #'lr_find_epochs': 1,
    #'lr_reduce_factor': 500,

    nn1_lr=1e-5
    nn1_max_lr=1e-2
    nn1_epochs = 30

    nn1_batch_size = 3
    nn1_num_workers = 2
    
    # Models as a list, maximium 3 items
    # These are settings that are used to create the NN1 model class
    # This is needed before loading class parameters and running inference.
    # Loading only works if the model class has been created
    nn1_models_class_generator = [{
    'class':'smp', #smp: segmentation models pytorch
    'arch': 'U_Net',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet', # TODO: support for using existing models (loading)
    'in_nchannels':1,
    'nclasses':2,
    }]

    nn1_axes_to_models_indices = [0,0,0] # By default use the same model for all axes
    # To use different models, use [0,1,2] for model0 along z, model1 along y, and model2 along x

    nn2_MLP_models_class_generator = {
        "nn2_hidden_layer_sizes" : "10,10",
        "nn2_activation": 'tanh',
        "nn2_out_nclasses": 2
    }

    #'learning_rate_init':0.001,
    #'solver':'sgd',
    nn2_max_iter = 1000
    nn2_ntrain = 262144 #Note that this is not a MLPClassifier parameter

    temp_data_outdir = None

# Define the MLP model
class MLPClassifier(nn.Module):
    # Should I add softmax?
    def __init__(self, input_size:int, hiden_sizes_list:list, output_size:int, activ_str:str):
        super().__init__()

        size0= input_size
        self.layers=[]
        for hls in hiden_sizes_list:
            hid_layer0 =  nn.Linear(size0, hls)
            self.layers.append(hid_layer0)
            size0=hls
        #last layer
        self.layers.append(nn.Linear(size0, output_size))

        if "tanh" in activ_str.lower():
            self.activ = nn.Tanh()
        elif "relu" in activ_str.lower():
            self.activ = nn.ReLU()
        elif "sigm" in activ_str.lower():
            self.activ = nn.Sigmoid()
        else:
            raise ValueError(f"activ_str {activ_str} not valid")

    def forward(self, x):
        for hl in self.layers:
            x= self.activ(hl(x))
        #x = self.sigm(x)
        return x
    
    def predict_class_as_cpu_np(self,x):
        p0 = self.forward(x)
        pred = torch.squeeze(torch.argmax(p0, dim=1))
        return pred.detach().cpu().numpy()

class cMultiAxisRotationsSegmentor2():

    def __init__(self, sett2: cSegmentor2_Settings=None):
        """
        Initialise an instance of cMultiAxisRotationsSegmentor2
        with default settings.

        if sett2 is None, initialises with basic setup

        If you wish different models for different axis, please use alternative helper creation functions

        """
        self.settings = sett2
        #self.cuda_device=cuda_device
        if sett2 is None:
            self.settings = cSegmentor2_Settings()

        self._NN1_models = None
        self._NN2_model = None

        self.init_models_from_settings()

        self.all_nn1_pred_pd=None
        self._tempdir_pred=None #TODO: Not sure this is needed


    def init_models_from_settings(self):
        self._NN1_models = [ self.create_nn1_ptmodel_from_class_generator(x).to(f"cuda:{self.settings.cuda_device}") for x in self.settings.nn1_models_class_generator]
        self._NN2_model = self.create_nn2_ptmodel_from_class_generator( self.settings.nn2_MLP_models_class_generator.to(f"cuda:{self.settings.cuda_device}") )
        
    @staticmethod
    def create_nn1_ptmodel_from_class_generator(nn1_cls_gen_dict: dict):
        # get segm model from dictionary item
        model0=None

        if nn1_cls_gen_dict['class'].lower()=='smp': #unet, AttentionNet (manet) and fpn
            #Segmentation models pytorch
            arch = nn1_cls_gen_dict['arch'].lower()
            if arch=="unet" or arch=="u_net":
                NN_class = smp.Unet
            elif arch=="manet":
                model0 = smp.MAnet
            elif arch=="fpn":
                model0 = smp.FPN
            else:
                raise ValueError(f"arch:{arch} not valid.")
            
            model0 = NN_class(
                encoder_name = nn1_cls_gen_dict['encoder_name'],
                encoder_weights = nn1_cls_gen_dict['encoder_weights'],
                in_channels = nn1_cls_gen_dict['in_nchannels'],
                classes = nn1_cls_gen_dict['nclasses'],
                activation = "sigmoid"
                )
        else:
            raise ValueError(f"class {nn1_cls_gen_dict['class']} not supported.")
        
        # TODO: add other 2D model support, not just SMPs

        return model0
    
    @staticmethod
    def create_nn2_ptmodel_from_class_generator(nn2_cls_gen_dict: dict):
        hid_layers = nn2_cls_gen_dict['nn2_hidden_layer_sizes'].split(",")

        if len(hid_layers)==0:
            ValueError(f"Invalid nn2_hidden_layer_sizes : {nn2_cls_gen_dict['nn2_hidden_layer_sizes']}")

        hid_layers_num_list = map(int, hid_layers)

        model0 = MLPClassifier(
            12,
            hid_layers_num_list,
            nn2_cls_gen_dict['nn2_out_nclasses'],
            nn2_cls_gen_dict["nn2_activation"]
            )
        
        return model0

    def set_cuda_device(self,n):
        #self._cuda_device=n
        self.settings.cuda_device=n


    @staticmethod
    def load_from_file(filepath):
        pass
        # TODO


    @staticmethod
    def create_simple_separate_models_per_axis(nclasses, cuda_device=0):
        """
        Sets up seperate smp unets for each of the axis
        """

        if nclasses<2:
            raise ValueError(f"nclasses {nclasses} too small")
        
        nn1_dict_gen = {'class':'smp', #smp: segmentation models pytorch
            'arch': 'U_Net',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet', # TODO: support for using existing models (loading)
            'in_nchannels':1, #greyscale
            'nclasses':nclasses,
        }

        nn2_MLP_models_class_generator = {
            "nn2_hidden_layer_sizes" : "10,10",
            "nn2_activation": 'tanh',
            "nn2_out_nclasses": nclasses
        }


        sett2 = cSegmentor2_Settings() #get default settings and modify

        sett2.nn1_models_class_generator = [nn1_dict_gen,
            nn1_dict_gen.copy(),
            nn1_dict_gen.copy()
        ]

        sett2.nn1_axes_to_models_indices = [0,1,2]

        sett2.nn2_MLP_models_class_generator = nn2_MLP_models_class_generator

        sett2.cuda_device = cuda_device #Probably should use kwargs

        #Create instance of cMultiAxisRotationsSegmentor2
        segm2 = cMultiAxisRotationsSegmentor2(sett2) # will run init()

        return segm2


    def train(self, traindata, trainlabels, get_metrics=True):
        """
        Train NN1 (volume segmantics) and NN2 (MLP Classifier)

        Returns:
            TODO
            
        """

        logging.debug(f"train()")
        trainlabels0 = None
        traindata0=None

        #Check traindata is 3D or list
        if isinstance(traindata, np.ndarray) and isinstance(trainlabels, np.ndarray) :
            logging.info("traindata and trainlabels are ndarray")

            #Convert to list so that can be used later
            traindata0 = [traindata]
            trainlabels0=[trainlabels]
        else:
            if isinstance(traindata, list) and isinstance(trainlabels, list):
                logging.info("traindata and trainlabels are list")
                if len(traindata)!=len(trainlabels):
                    raise ValueError("len(traindata)!=len(trainlabels) error. Must be the same number of items.")
                
                traindata0=traindata
                trainlabels0=trainlabels
        
        #Check dimensions of volumes
        traindata_ndims = [x.ndim for x in traindata0]
        trainlabels_ndims = [x.ndim for x in trainlabels0]
        logging.info(f"traindata_ndims:{traindata_ndims}")
        logging.info(f"trainlabels_ndims:{trainlabels_ndims}")

        if np.any(traindata_ndims!=3) or np.any(trainlabels_ndims!=3):
            raise ValueError(f"traindata or trainlabels not 3D")

        self.labels_dtype= trainlabels[0].dtype

        #How many sets?
        nsets=len(traindata0)
        logging.info(f"nsets:{nsets}")


        # ** Train NN1
        self.NN1_train(traindata0, trainlabels0)
        #(This does not return anything.)


        #TODO *******

        # ** Predict NN1
        #Does the multi-axis multi-rotation predictions
        # and collects data files
        
        # Setup temporary folders to store predictions
        self._tempdir_pred=None
        if self.temp_data_outdir is None:
            self._tempdir_pred= tempfile.TemporaryDirectory()
            tempdir_pred_path = Path(self._tempdir_pred.name)
        else:
            tempdir_pred_path=Path(self.temp_data_outdir)
        
        logging.info(f"tempdir_pred_path:{tempdir_pred_path}")

        #Predict multi-axis multi-rotations
        #Predictions are stored in h5 files in temporary folder

        self.all_nn1_pred_pd = self.NN1_predict(traindata0, tempdir_pred_path) #note that 
        logging.info("NN1_predict returned")
        logging.info(self.all_nn1_pred_pd)


        #TODO
        #Take this oportunity to calculate metrics of each prediction labels if required
        # nn1_acc_dice_s= []
        # #pred_data_probs_filenames=all_pred_pd['pred_data_labels_filenames'].tolist() #note that all sets will be included in this list
        # if get_metrics:
        #     logging.info("Collecting NN1 metrics")
        #     for i, prow in self.all_nn1_pred_pd.iterrows():
        #         pred_labels_fn = prow['pred_data_labels_filenames']
        #         iset = prow['pred_sets']
        #         ipred = prow['pred_ipred']
        #         data_i = read_h5_to_np(pred_labels_fn)

        #         #What is the corresponding iset?
        #         a0 =  metrics.MetricScoreOfVols_Accuracy(data_i,trainlabels0[iset])
        #         d0 = metrics.MetricScoreOfVols_Dice(data_i,trainlabels0[iset])
        #         nn1_acc_dice_s.append( [a0,d0])
        #         logging.info(f"prediction iset:{iset}, ipred:{ipred}, filename: {pred_labels_fn}, accuracy:{a0}, dice:{d0}")

        #     #add metrics to pandas dataframe with results
        #     acc_list0 = [ ad0[0] for ad0 in nn1_acc_dice_s]
        #     self.all_nn1_pred_pd["accuracy"]= acc_list0

        #     dice_list0 = [ ad0[1][0] for ad0 in nn1_acc_dice_s]
        #     self.all_nn1_pred_pd["dice"]= dice_list0



        # ** NN2 training

        #Need to train next model by running predictions and optimize MLP
        #Use multi-predicted data and labels to train NN2
        
        #Build data object containing all predictions (5D - iset, ipred, z,y,x, class)
        npredictions_per_set = int(np.max(self.all_nn1_pred_pd['pred_ipred'].to_numpy())+1)
        logging.info(f"npredictions_per_set:{npredictions_per_set}")
        data_all_np5d=None

        logging.debug("Aggregating multiple sets onto a single volume data_all_np5d")
        # aggregate multiple sets for data
        for i,prow in self.all_nn1_pred_pd.iterrows():

            prob_filename = prow['pred_data_probs_filenames']
            data0 = read_h5_to_np(prob_filename)

            if i==0:
                #initialise
                logging.info(f"data0.shape:{data0.shape}")
                all_shape0 = (
                    nsets,
                    npredictions_per_set,
                    *data0.shape
                    )
                # (iset, ipred, iz,iy,ix, ilabel) , 5dim

                data_all_np5d=np.zeros( all_shape0 , dtype=data0.dtype)

            
            ipred=prow['pred_ipred']
            iset=prow['pred_sets']

            data_all_np5d[iset,ipred, :,:,:, :] = data0


        #Train NN2 from multi-axis multi-angle predictions against labels (gnd truth)
        nn2_acc, nn2_dice = self.NN2_train(data_all_np5d, trainlabels0, get_metrics=get_metrics)

        #Preserve for debugging
        # if not tempdir_pred is None:
        #     tempdir_pred.cleanup()

        #return nn1_acc_dice_s, (nn2_acc, nn2_dice)

        return None
    
    
    def predict(self, data_in, use_dask=False):
        """
        Creates predicted labels from a whole data volume
        using the double NN1+NN2 pipeline
        """
        #Predict from provided volumetric data using the trained model defined here

        #Check if the following objects are avaialble
        #self.volseg2pred #NN1 predictor (attention the NN1_predict() loads the model from file!!)
        logging.debug(f"predict() data_in.shape:{data_in.shape}, data_in.dtype:{data_in.dtype}, use_dask:{use_dask}")

        if not self.model_NN1_path is None and not self.NN2 is None:
            logging.info("Setting up NN1 prediction")

            self._tempdir_pred=None
            if self.temp_data_outdir is None:
                self._tempdir_pred= tempfile.TemporaryDirectory()
                tempdir_pred_path = Path(self._tempdir_pred.name)
            else:
                tempdir_pred_path=Path(self.temp_data_outdir)

            #pred_data_probs_filenames, _ = self.NN1_predict(data_in, tempdir_pred_path) #Get prediction probs, not labels
            self.all_nn1_pred_pd = self.NN1_predict(data_in, tempdir_pred_path) #Get prediction probs, not labels
            logging.info("NN1 prediction, complete.")
            logging.info("all_pred_pd")
            logging.info(self.all_nn1_pred_pd)
            
            data_all = self.aggregate_nn1_pred_data(use_dask)

            if data_all is None:
                logging.error("After aggregation, data_all is None")
            
            d_prediction=None #Default return value
            if not data_all is None:
                logging.info("Setting up NN2 prediction")
                d_prediction= self.NN2_predict(data_all)
                
                logging.info("NN2 prediction complete.")

            # if not tempdir_pred is None:
            #     logging.info(f"Cleaning up tempdir_pred: {tempdir_pred_path}")
            #     tempdir_pred.cleanup()

            return d_prediction


    def NN1_train(self, traindata_list, trainlabels_list):
        """
        traindata: a list of 3d volumes
        trainlabels : a list of 3d volumes with corresponding labels

        This code is share similarities to volumesegmantics train_2d_model.py
        
        """

        logging.info("NN1_train()")

        if not(isinstance(traindata_list, list) and isinstance(trainlabels_list, list) ):
            raise ValueError("Invalid traindata_list or trainlabels_list")

        idx_models = np.unique(self.settings.nn1_axes_to_models_indices)
        logging.info(f"nmodels:{idx_models}")

        # Setup augmentations to apply to 2D images
        logging.info("Setting up augmentations")
        

        logging.info("Preprocess volume according to data_vol_norm_process")
        if not self.settings.data_vol_norm_process is None:

            #Normalise volumetric data to setting chosen
            traindata_list0=[]

            if "mean_stdev_3" in self.settings.data_vol_norm_process:
                # Clip data to -3*stdev and +3*stdev and normalises to values between 0 and 1
                for d0 in traindata_list:
                    d0_mean = np.mean(d0)
                    d0_std = np.std(d0)

                    if d0_std==0:
                        raise ValueError("Error. Stdev of data volume is zero.")
                    
                    d0_corr = (d0.astype(np.float32) - d0_mean) / d0_std
                    d0_corr = (np.clip(d0_corr, -3.0, 3.0) +3.0) / 6.0
                    
                    traindata_list0.append(d0_corr*255)

            elif "mean_stdev_3_5" in self.settings.data_vol_norm_process:
                for d0 in traindata_list:
                    d0_mean = np.mean(d0)
                    d0_std = np.std(d0)

                    if d0_std==0:
                        raise ValueError("Error. Stdev of data volume is zero.")
                    
                    d0_corr = (d0.astype(np.float32) - d0_mean) / d0_std
                    d0_corr = (np.clip(d0_corr, -3.0, 5.0) +3.0) / 8.0

                    traindata_list0.append(d0_corr*255)
            
            #replace traindata_list with corrected
            traindata_list = traindata_list0

        #Ensure data is uint8
        traindata_list = [ t.astype(np.uint8) for t in traindata_list]

        logging.info("Creating train and validation dataloaders for each model")
        # Dataloader(s) will depend on the number of models and respective axes
        dataloaders_train=[]
        dataloaders_test=[]
        for i in idx_models:
            #Gets the axes that the NN1 model is supposed to be used
            model_axes= np.flatnonzero(
                np.array(self.settings.nn1_axes_to_models_indices) == i
            ).tolist()

            # Get shape along the axis

            ds0 = NN1_train_input_dataset_along_axes(
                traindata_list,
                trainlabels_list,
                model_axes,
                self.settings.cuda_device
            )

            dset1, dset2 = torch.utils.data.random_split(ds0, [0.8,0.2])

            dl_train = DataLoader(dset1, batch_size=self.settings.nn1_batch_size, shuffle=True)
            dl_test = DataLoader(dset2, batch_size=self.settings.nn1_batch_size, shuffle=True)
            logging.info(f"Train and test dataloaders created for model number:{i}, model_axes:{model_axes}, len(train):{len(dl_train)}, len(test):{len(dl_test)}")

            dataloaders_train.append(dl_train)
            dataloaders_test.append(dl_test)

        logging.info("All dataloaders created.")
        logging.info(f"len(dataloaders_train): {len(dataloaders_train)}")
        logging.info(f"len(dataloaders_test): {len(dataloaders_test)}")

        
        # Setup losses
        # Note that output from SMP is already sigmoided, hence the 'logits' versions of losses are not used
        nn1_loss_func = None
        if "celoss" in self.settings.nn1_loss_criterion.lower():
            nn1_loss_func = torch.nn.CrossEntropyLoss().to('cuda')
            #nn1_loss_func = smp.losses.SoftCrossEntropyLoss().to('cuda')
        elif "diceloss" in self.settings.nn1_loss_criterion.lower():
            nn1_loss_func = smp.losses.DiceLoss(mode='multiclass', from_logits=False).to('cuda')
        else:
            raise ValueError(f"{self.settings.nn1_loss_criterion} not a valid loss criteria")
        
        # Setup metrics for test data
        # TODO
        nn1_metric_func = None
        if "meaniou" in self.settings.nn1_eval_metric.lower():
            nn1_metric_func = segmentation_models_pytorch.utils.metrics.IoU()
        elif "dice" in self.settings.nn1_eval_metric.lower() or "fscore" in self.settings.nn1_eval_metric.lower():
            nn1_metric_func = segmentation_models_pytorch.utils.metrics.Fscore()
        elif "accuracy" in self.settings.nn1_eval_metric.lower():
            nn1_metric_func = segmentation_models_pytorch.utils.metrics.Accuracy()


        #Train each model
        for i,model0 in enumerate(self._NN1_models):
            # get respective dataloader
            dl_train0 = dataloaders_train[i]
            dl_test0 = dataloaders_test[i]

            #Setup optimizer and scaler

            optimizer = torch.optim.AdamW(model0.parameters(), lr=self.settings.nn1_lr)
            scaler=torch.cuda.amp.GradScaler()

            epochs = self.settings.nn1_epochs

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr= self.settings.nn1_max_lr,
                steps_per_epoch=len(dl_train0),
                epochs=epochs,
                #pct_start=0.1, #default=0.3
                )
            
            train_model(model0, dl_train0, dl_test0, nn1_loss_func, optimizer, scaler, scheduler,
                        epochs=epochs,
                        metric_fn=nn1_metric_func
                        )

            logging.info(f"Training model {i} complete")

        logging.info("All models trained successfully. Don't forget to save them.")



    def NN1_predict(self,data_to_predict, pred_folder_out):
        """
        
        Does the multi-axis multi-rotation predictions
        and returns predictions filenames of probablilities and labels

        predictions are probabilities (not labels)

        Params:
            data_to_predict: a ndarray or a list of ndarrays with the 3D data to rund predictions from
            pred_folder_out: a string with the location of where to drop results in h5 file format


        Returns:
            a pandas Dataframe with results of predictions in
            filenames of probabilities and labels,
            and respective set, rotation, plane, and ipred

            Columns are
                'pred_data_probs_filenames'
                'pred_data_labels_filenames'
                'pred_sets'
                'pred_planes'
                'pred_rots'
                'pred_ipred'

        """

        logging.info("NN1_predict()")
        #Internal functions
        def _save_pred_data(data, count,axis, rot):
            # Saves predicted data to h5 file in tempdir and return file path in case it is needed
            file_path = f"{pred_folder_out}/pred_{count}_{axis}_{rot}.h5"
            
            utils.save_data_to_hdf5(data, file_path)
            return file_path

        # Ensure data_to_predict_l is a list of datasets
        data_to_predict_l=None
        if not isinstance(data_to_predict, list):
            logging.debug("data_to_predict not a list. Converting to list")
            data_to_predict_l=[data_to_predict]
        else:
            logging.debug("data_to_predict is a list. No conversion needed")
            data_to_predict_l=data_to_predict

        pred_data_probs_filenames=[] #Will store results in files, and keep the filenames as reference
        pred_data_labels_filenames=[]
        pred_sets=[]
        pred_planes=[]
        pred_rots=[]
        pred_ipred=[]
        pred_shapes=[]

        logging.info(f"number of data sets to predict: {len(data_to_predict_l)}")
        
        for iset, data_to_predict0 in enumerate(data_to_predict_l):
            logging.info(f"Data to predict index:{iset}")
            data_vol0 = np.array(data_to_predict0) #Copies

            #setup Prediction Manager
            #It will also clip data depending on settings, and to get that data
            # it is property data_vol
            # volseg2pred_m = VolSeg2DPredictionManager(
            #     model_file_path= self.model_NN1_path,
            #     data_vol=data_vol1,
            #     settings=self.NN1_pred_settings,
            #     #use_dask=True
            #     )

            #Create dataloaders along different axis for this specific data_to_predict0
            # Input_dataset_along_axes

            itag=0

            for krot in range(0, 4):
                rot_angle_degrees = krot * 90
                logging.info(f"Volume to be rotated by {rot_angle_degrees} degrees")

                #Predict 3 axis
                #YX, along Z
                planeYX=(1,2)
                logging.info("Predicting YX slices, along Z")
                data_vol = np.rot90(np.array(data_vol0),krot, axes=planeYX) #rotate

                res = self.nn1_predict_slices_along_axis(data_vol, axis=0)

                #invert rotations before saving
                pred_probs = np.rot90(res[1], -krot, axes=planeYX) 
                pred_labels = np.rot90(res[0], -krot, axes=planeYX)

                fn = _save_pred_data(pred_probs, iset, "YX", rot_angle_degrees)
                fn = _save_pred_data(pred_labels, iset, "YX_labels", rot_angle_degrees)

                pred_data_labels_filenames.append(fn)
                pred_data_probs_filenames.append(fn)
                pred_sets.append(iset)
                pred_planes.append("YX")
                pred_rots.append(rot_angle_degrees)
                pred_ipred.append(itag)
                pred_shapes.append(pred_labels.shape)
                itag+=1

                
                #ZX
                logging.info("Predicting ZX slices, along Y")
                planeZX=(0,2)
                data_vol = np.rot90(np.array(data_vol0),krot, axes=planeZX) #rotate
                
                res = self.nn1_predict_slices_along_axis(data_vol, axis=1)

                pred_probs = np.rot90(res[1], -krot, axes=planeZX) #invert rotation before saving
                pred_labels = np.rot90(res[0], -krot, axes=planeZX)

                fn = _save_pred_data(pred_probs, iset, "ZX", rot_angle_degrees)
                fn = _save_pred_data(pred_labels, iset, "ZX_labels", rot_angle_degrees)

                pred_data_labels_filenames.append(fn)
                pred_data_probs_filenames.append(fn)
                pred_sets.append(iset)
                pred_planes.append("ZX")
                pred_rots.append(rot_angle_degrees)
                pred_ipred.append(itag)
                pred_shapes.append(pred_labels.shape)
                itag+=1

                #ZY
                logging.info("Predicting ZY slices, along X")
                planeZY=(0,1)
                data_vol = np.rot90(np.array(data_vol0),krot, axes=planeZY) #rotate
                
                res = self.nn1_predict_slices_along_axis(data_vol, axis=2)

                pred_probs = np.rot90(res[1], -krot, axes=planeZY) #invert rotation before saving
                pred_labels = np.rot90(res[0], -krot, axes=planeZY)
                fn = _save_pred_data(pred_probs, iset, "ZY", rot_angle_degrees)
                fn = _save_pred_data(pred_labels, iset, "ZY_labels", rot_angle_degrees)

                pred_data_labels_filenames.append(fn)
                pred_data_probs_filenames.append(fn)
                pred_sets.append(iset)
                pred_planes.append("ZY")
                pred_rots.append(rot_angle_degrees)
                pred_ipred.append(itag)
                pred_shapes.append(pred_labels.shape)
                itag+=1

            del(data_vol0)

        logging.info("Generating a DataFrame object with information about predictions")

        all_pred_pd = pd.DataFrame({
            'pred_data_probs_filenames': pred_data_probs_filenames,
            'pred_data_labels_filenames': pred_data_labels_filenames,
            'pred_sets':pred_sets,
            'pred_planes':pred_planes,
            'pred_rots':pred_rots,
            'pred_ipred':pred_ipred,
            'pred_shapes': pred_shapes,
        })
        
        logging.info("NN1_predict() complete")

        return all_pred_pd


    def NN2_train(self, train_data_all_probs_5d, trainlabels_list, get_metrics=True):
        logging.debug("NN2 train()")

        #Assumes train_data_all_probs_list is 5d
        # and that trainlabels_list is a list of 3d volumes

        assert train_data_all_probs_5d.shape[0]==len(trainlabels_list)

        nsets= len(trainlabels_list)

        logging.debug("Getting several points to train NN2")
        # #This is probably not the best way to get a random points
        # #Get several points to train NN2
        # x_origs = np.arange(0, train_data_all_probs_5d.shape[3],5)
        # y_origs = np.arange(0,train_data_all_probs_5d.shape[2],5)
        # z_origs = np.arange(0,train_data_all_probs_5d.shape[1],5)
        # x_mg, y_mg, z_mg = np.meshgrid(x_origs,y_origs, z_origs)
        # all_origs_list = np.transpose(np.vstack( (z_mg.flatten() , y_mg.flatten() , x_mg.flatten() ) ) ).tolist()

        # random.shuffle(all_origs_list)
        # #ntrain = min(len(all_origs_list), 4096)
        # ntrain = min(len(all_origs_list), self.NN2_settings.ntrain)

        # X_train=[] # as list of volume data, flattened for each voxel
        
        # iset_randoms = np.random.default_rng().integers(0,nsets,ntrain)

        # for i in tqdm.trange(ntrain):
        #     el = all_origs_list[i]
        #     z,y,x = el
        #     data_vol = train_data_all_probs_5d[iset_randoms[i],:,z,y,x,:]
        #     data_vol_flat = data_vol.flatten()
        #     X_train.append(data_vol_flat)

        # y_train=[] # labels
        # for i in tqdm.trange(ntrain):
        #     el = all_origs_list[i]
        #     z,y,x = el
        #     label_vol_label = trainlabels_list[iset_randoms[i]][z,y,x]
        #     y_train.append(label_vol_label)

        ntrain=self.NN2_settings.ntrain

        iset_rnd = np.random.randint(0,nsets,ntrain)
        z_orig_rnd = np.random.randint(0,train_data_all_probs_5d.shape[2],ntrain)
        y_orig_rnd = np.random.randint(0,train_data_all_probs_5d.shape[3],ntrain)
        x_orig_rnd = np.random.randint(0,train_data_all_probs_5d.shape[4],ntrain)
        
        all_origs_list=np.column_stack( (iset_rnd,z_orig_rnd,y_orig_rnd,x_orig_rnd ))

        #Could probably check for duplicates, but I will sckip that part
        #Collect voxels data
        X_train=[]
        Y_train=[] # labels
        for i in tqdm.trange(ntrain):
            el = all_origs_list[i,:]
            iset,z,y,x = el
            data_vol = train_data_all_probs_5d[iset,:,z,y,x,:]
            data_vol_flat = data_vol.flatten()
            X_train.append(data_vol_flat)

            label_vol_label = trainlabels_list[iset][z,y,x]
            Y_train.append(label_vol_label)

        logging.debug(f"NN2 len(X_train):{len(X_train)} , len(Y_train):{len(Y_train)}")

        #Setup classifier
        logging.info("Setup NN2 MLPClassifier")
        #self.NN2 = MLPClassifier(hidden_layer_sizes=(10,10), random_state=1, activation='tanh', verbose=True, learning_rate_init=0.001,solver='sgd', max_iter=1000)
        #self.NN2 = MLPClassifier(**self.NN2_settings.__dict__) #Unpack dict to become parameters

        # TODO: torch based MLP
        # self.NN2 = MLPClassifier(
        #     hidden_layer_sizes=self.NN2_settings.hidden_layer_sizes,
        #     activation=self.NN2_settings.activation,
        #     random_state=self.NN2_settings.random_state,
        #     verbose=self.NN2_settings.verbose,
        #     learning_rate_init=self.NN2_settings.learning_rate_init,
        #     solver=self.NN2_settings.solver,
        #     max_iter=self.NN2_settings.max_iter
        #     )
        # #Do the training here
        # logging.info(f"NN2 MLPClassifier fit with {len(X_train)} samples, (y_train {len(Y_train)} samples)")
        # self.NN2.fit(X_train,Y_train)



        logging.info(f"NN2 train score:{self.NN2.score(X_train,Y_train)}")

        nn2_acc=[]
        nn2_dice=[]
        if get_metrics:
            logging.info("Preparing to predict the whole training volume")

            for i in range(nsets):
                d_prediction= self.NN2_predict( train_data_all_probs_5d[i,:,:,:,:,:])

                #Get metrics
                nn2_acc0= metrics.MetricScoreOfVols_Accuracy(trainlabels_list[i],d_prediction)
                nn2_dice0= metrics.MetricScoreOfVols_Dice(trainlabels_list[i],d_prediction, useBckgnd=False)

                logging.info(f"set {i}, NN2 acc:{nn2_acc0}, dice:{nn2_dice0}")
                nn2_acc.append(nn2_acc0)
                nn2_dice.append(nn2_dice0)
        
        return nn2_acc, nn2_dice
    

    def NN2_predict(self, data_all_probs):
        # version that uses ParallelPostfit
        logging.debug("NN2_predict()")
        from dask_ml.wrappers import ParallelPostFit
        from dask.diagnostics import ProgressBar

        data_all_probs_da=None
        if isinstance(data_all_probs, np.ndarray):
            logging.info("Data type is numpy.ndarray")
            data_all_probs_da = da.from_array(data_all_probs)
        
        elif isinstance(data_all_probs, da.core.Array):
            logging.info("Data type is dask.core.Array")
            #Use dask reduction functionality to do the predictions
            data_all_probs_da=data_all_probs
        
        if data_all_probs_da is None:
            raise ValueError("data_all_probs invalid")
        

        #Need to flatten along the npred and nclasses
        data_2MLP_t= da.transpose(data_all_probs_da,(1,2,3,0,4))

        dsize = data_2MLP_t.shape[0]*data_2MLP_t.shape[1]*data_2MLP_t.shape[2]
        inputsize = data_2MLP_t.shape[3]*data_2MLP_t.shape[4]

        data_2MLP_t_reshape = da.reshape(data_2MLP_t, (dsize, inputsize))

        mlp_PPF_parallel = ParallelPostFit(self.NN2)
        mlppred = mlp_PPF_parallel.predict(data_2MLP_t_reshape)

        #Reshape back to 3D
        mlppred_3D = da.reshape(mlppred, data_2MLP_t.shape[0:3])
        
        # logging.info("Starting NN2 predict dask computation")
        pbar = ProgressBar()
        with pbar:
            b_comp=mlppred_3D.compute() #compute and convert to numpy

        return b_comp


    def save_model(self, filename):
        """
        Saves model to a zip file containing the following
        NN1 model (volume segmantics pytorch)
        NN1 settings (yaml file?)
        NN2 model (MPL pickle)
        NN2 settings
        """

        #Generate files in temporary storage
        #tempdir_model = tempfile.TemporaryDirectory()
        #tempdir_model_path=Path(tempdir_model)

        logging.debug("save_model()")

        import io
        import joblib
        #import pickle
        
        #NN1 settings
        nn1_train_settings_bytesio = io.BytesIO()
        joblib.dump(self.NN1_train_settings, nn1_train_settings_bytesio)

        nn1_pred_settings_bytesio = io.BytesIO()
        joblib.dump(self.NN1_pred_settings, nn1_pred_settings_bytesio)

        # nn2_settings_bytesio = io.BytesIO()
        # joblib.dump(self.NN2_settings, nn2_settings_bytesio)
        # Don't need to save NN2 settings seperately
        # as they are already included in NN2 MLPclassifier (self.NN2)

        #NN2 model
        nn2_model_bytesio = io.BytesIO()
        joblib.dump(self.NN2, nn2_model_bytesio)

        #NN1 model is in file path self.model_NN1_path

        from zipfile import ZipFile

        with ZipFile(filename, 'w') as zipobj:
            zipobj.write(str(self.model_NN1_path), arcname="NN1_model.pytorch")
            zipobj.writestr("NN1_train_settings.joblib",nn1_train_settings_bytesio.getvalue())
            zipobj.writestr("NN1_pred_settings.joblib", nn1_pred_settings_bytesio.getvalue())
            zipobj.writestr("NN2_model.joblib", nn2_model_bytesio.getvalue())


        nn1_pred_settings_bytesio.close()
        nn1_pred_settings_bytesio.close()
        nn2_model_bytesio.close()

    #Do not save the pandas file

    def load_model(self, filename):
        #import io
        logging.debug("load_model()")
        import joblib
        from zipfile import ZipFile

        with ZipFile(filename, 'r') as zipobj:
            ##NN1 model
            self._nn1_model_temp_dir = tempfile.TemporaryDirectory()
            zipobj.extract("NN1_model.pytorch",self._nn1_model_temp_dir.name)
            self.model_NN1_path=Path(self._nn1_model_temp_dir.name,"NN1_model.pytorch")

            with zipobj.open("NN1_train_settings.joblib",'r') as z0:
                self.NN1_train_settings= joblib.load(z0)

            with zipobj.open("NN1_pred_settings.joblib",'r') as z1:
                self.NN1_pred_settings= joblib.load(z1)
            
            with zipobj.open("NN2_model.joblib",'r') as z2:
                self.NN2= joblib.load(z2)

    @staticmethod
    def create_from_model( filename):
        newobj = cMultiAxisRotationsSegmentor()
        newobj.load_model(filename)

        return newobj
    
    def aggregate_nn1_pred_data(self, use_dask=False):
        logging.debug(f"aggregate_nn1_pred_data with use_dask:{use_dask}")
        if self.all_nn1_pred_pd is None:
            return None
        
        logging.info("Building large object containing all predictions.")
        #Build data object containing all predictions
        #Try using numpy. If memory error use dask instead

        data_all=None

        if not use_dask:
            logging.info("use_dask=False. Will try to aggregate data to a numpy.ndarray")
            try:
                data_all=None
                # aggregate multiple sets for data
                for i,prow in tqdm.tqdm(self.all_nn1_pred_pd.iterrows(), total=self.all_nn1_pred_pd.shape[0]):

                    prob_filename = prow['pred_data_probs_filenames']
                    data0 = read_h5_to_np(prob_filename)

                    if i==0:
                        #initialise
                        logging.info(f"data0.shape:{data0.shape}")
                        npredictions = int(np.max(self.all_nn1_pred_p['pred_ipred'].to_numpy())+1)
                        logging.info(f"npredictions:{npredictions}")
                        
                        all_shape = (
                            npredictions,
                            *data0.shape
                            )
                        # (ipred, iz,iy,ix, ilabel) , 5dim
                        
                        data_all = np.zeros(all_shape, dtype=data0.dtype)

                    data_all[i,:,:,:,:]=data0

            except Exception as exc0:
                logging.info("Allocation using numpy failed. Failsafe will use dask.")
                logging.info(f"Exception type:{type(exc0)}")
                use_dask=True
            
        if use_dask:
            logging.info("use_dask=True. Will aggregate data to a dask.array object")
            try:
                data_all=None
                # aggregate multiple sets for data
                for i,prow in tqdm.tqdm(self.all_nn1_pred_pd.iterrows(), total=self.all_nn1_pred_pd.shape[0]):

                    prob_filename = prow['pred_data_probs_filenames']
                    data0 = read_h5_to_da(prob_filename)

                    if i==0:
                        #initialise
                        logging.info(f"i:{i}, data0.shape:{data0.shape}, data0.chunksize:{data0.chunksize} ")
                        npredictions = int(np.max(self.all_nn1_pred_pd['pred_ipred'].to_numpy())+1)
                        logging.info(f"npredictions:{npredictions}")
                        
                        #chunks_shape = (npredictions, *data0.chunksize )
                        #in case of 12 predictions and 3 labels, the chunks will be (12,128,128,128,3) size
                        all_shape = ( npredictions,*data0.shape)

                        #max chunksize in xyz of 1024
                        zyx_chunks_orig= data0.chunksize[:-1]
                        zyx_chunks_max= [ min(s,1024) for s in zyx_chunks_orig ]
                        chunks_shape = ( npredictions,*zyx_chunks_max,data0. chunksize[-1] )

                        logging.info(f"data_all shape:{all_shape} chunks_shape:{chunks_shape}")

                        # (ipred, iz,iy,ix, ilabel) , 5dim
                        data_all=da.zeros(all_shape, chunks=chunks_shape , dtype=data0.dtype)

                    data_all[i,:,:,:,:]=data0

                bcomplete=True
            except Exception as exc0:
                logging.info("Allocation failed with dask. Returning None")
                logging.info(f"Exception type:{type(exc0)}")
                logging.info("Exception string:", str(exc0))
                data_all=None
        
        return data_all
    
    def NN1_predict_standard(self,data_vol, pred_file_h5_out):
        """
        Does the volume segmantics predictions using its 'standard' way, with a single volume output
        
        Params:
            data_vol: path to file or ndarray. Single, not a list
            pred_file_h5_out: a string with the location of where to drop results in h5 file format *.h5

        Returns:
            nothing. Output result should be saved in filename pred_file_h5_out
        
        """

        logging.debug("NN1_predict_standard()")

        if self.model_NN1_path is None:
            logging.error("self.model_NN1_path is None. Exiting")
            return None

        logging.info(f"pred_file_h5_out: {pred_file_h5_out}")

        volseg2pred_m = VolSeg2DPredictionManager(
                model_file_path= self.model_NN1_path,
                data_vol=data_vol,
                settings=self.NN1_pred_settings,
                #use_dask=True
                )
        
        pred_file_h5_out_path = Path(pred_file_h5_out)
        volseg2pred_m.predict_volume_to_path(pred_file_h5_out_path)

        logging.info(f"Prediction completed, saved to file: {pred_file_h5_out}")

    @staticmethod
    def _squeeze_merge_vols_by_max_prob( probs2, labels2):
        #Code from volumesegmantics that merges predictions
        logging.debug("_merge_vols_by_max_prob()")
        max_prob_idx = np.argmax(probs2, axis=0)
        max_prob_idx = max_prob_idx[np.newaxis, :, :, :]
        probs2[0] = np.squeeze(
            np.take_along_axis(probs2, max_prob_idx, axis=0)
        )
        labels2[0] = np.squeeze(
            np.take_along_axis(labels2, max_prob_idx, axis=0)
        )
        return
        

    def NN1_predict_extra_from_last_prediction(self, do_vspred=False, do_cs=False):
        """
        Does the volume-segmantics predictions and consistency score calcualation
        and returns a dictionary with the results.
        Uses the temporary output files from last prediction to work out the new prediction.

        If trying to do vspred and consistency score at the same time and this routine uses too much RAM
        try to execute one at the time

        TODO: option to use dask for handling data
        
        Params:
            do_vspred: calculate the predictions in 'standard' colume-segmantics way
            do_cs: calculate consistency score using the probabilities

        Returns:
            A dictionary
                'vspred' : predicted volume-segmantics
                'cs': the consistency score volume
        """

        logging.debug("NN1_predict_extra_from_last_prediction()")

        labels_vs_2stack=None
        probs_vs_2stack=None

        if do_cs: 
            from . import ConsistencyScore
            consistencyscore0 = ConsistencyScore.cConsistencyScoreMultipleWayProbsAccumulate()

        # Uses pandas list created with location of files
        for i,prow in tqdm.tqdm(self.all_nn1_pred_pd.iterrows(), total=self.all_nn1_pred_pd.shape[0]):

            if do_vspred or do_cs:
                pred_prob_filename = prow['pred_data_probs_filenames']
                pred_probs = read_h5_to_np(pred_prob_filename)

            if do_vspred:
                pred_labels_filename = prow['pred_data_labels_filenames']
                pred_labels = read_h5_to_np(pred_labels_filename)

                #Squeeze all probabilities along class dimension to maximum
                probs_class_squeezed = np.max(pred_probs, axis=pred_probs.ndim-1)

                if labels_vs_2stack is None:
                    logging.debug("First labels and probs file initializes")
                    shape_tup = pred_labels.shape
                    labels_vs_2stack = np.empty((2, *shape_tup), dtype=pred_labels.dtype)
                    probs_vs_2stack = np.empty((2, *shape_tup), dtype=pred_probs.dtype)
                    labels_vs_2stack[0]=pred_labels
                    probs_vs_2stack[0]=probs_class_squeezed
                else:
                    labels_vs_2stack[1]=pred_labels
                    probs_vs_2stack[1]=probs_class_squeezed
                    self._squeeze_merge_vols_by_max_prob(probs_vs_2stack,labels_vs_2stack)
            
            if do_cs:
                consistencyscore0.accumulate(pred_probs)
        
        vspred=None
        cs=None

        if do_vspred:
            vspred=labels_vs_2stack[0]

        if do_cs:
            cs = consistencyscore0.getCScore()
        
        return {'cs':cs, 'vspred':vspred}



    @staticmethod
    def copy_from(lgsegm0):
        """ create new lgsegmentor object with properties being a copy of
        existing lgsegmentor object
        """
        import copy

        lgsegm1 = cMultiAxisRotationsSegmentor()

        lgsegm1.model_NN1_path = lgsegm0.model_NN1_path
        lgsegm1.chunkwidth = lgsegm0.chunkwidth 
        lgsegm1.nlabels=lgsegm0.chunkwidth 
        lgsegm1.temp_data_outdir=lgsegm0.temp_data_outdir
        lgsegm1.cuda_device=lgsegm0.cuda_device

        lgsegm1.NN1_train_settings = copy.deepcopy(lgsegm0.NN1_train_settings)
        lgsegm1.NN1_pred_settings = copy.deepcopy(lgsegm0.NN1_pred_settings)

        lgsegm1.NN2_settings = copy.deepcopy(lgsegm0.NN2_settings)

        lgsegm1.labels_dtype = lgsegm0.labels_dtype

        lgsegm1.all_nn1_pred_pd=None
        if not lgsegm0.all_nn1_pred_pd is None:
            lgsegm1.all_nn1_pred_pd = lgsegm0.all_nn1_pred_pd.copy()

        lgsegm1._nn1_model_temp_dir = lgsegm0.nn1_model_temp_dir
        lgsegm1.model_NN1_path= lgsegm0.model_NN1_path

        
        lgsegm1.NN2= copy.deepcopy(lgsegm0)

        return lgsegm1


    def dice_loss_np(y_true, y_pred): #old, not used anymore
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)
        return 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

    def cleanup(self):
        try:
            self._tempdir_pred.cleanup()
        except:
            pass

        try:
            self._pytorch_model_tempdir.cleanup()
        except:
            pass
        
        try:
            self._nn1_model_temp_dir.cleanup()
        except:
            pass

    #Context manager, to clean resources such as tempfiles efficiently
    # and provide a way for using `with` statements
    def __exit__(self, *args):
        self.cleanup()
    
    def __enter__(self):
        #Required to run with `with`
        return self

    def nn1_predict_slices_along_axis(self, datavol, axis):
        ds0 = VolumeSlicerDataset(datavol, axis , per_slice_tfms=None) #TODO: check per_slice_tfms
        dl0 = DataLoader(ds0, self.settings.nn1_batch_size, self.settings.nn1_num_workers)

        # Get correct model
        model_index = self.settings.nn1_axes_to_models_indices[axis]
        model = self._NN1_models[model_index]
        preds_list = []
        labels_list = []
        for x in dl0:
            X=X_parse(x)
            #y=y_parse(y)
            pred_probs_slice = model(X)
            preds_list.append(pred_probs_slice)

            # get labels using argmax
            lbl_slice = torch.argmax(pred_probs_slice, dim=1)
            labels_list.append(lbl_slice)

        preds = torch.cat(preds_list, dim=axis)
        labels = torch.cat(labels_list, dim=axis)

        return labels,preds


class NN1_train_input_dataset_along_axes(Dataset):
    def __init__(self, datavols_list, labelsvols_list, axes=[0,1,2], cuda_device=0):
        
        self.datavols = []
        self.labelsvols = []
        self.tfms=[]

        for d0, l0 in zip(datavols_list,labelsvols_list):

            #Preprocess? TODO

            for ax0 in axes:
                if ax0==0:
                    self.datavols.append(d0)
                    self.labelsvols.append(l0)
                    self.tfms.append(
                        get_train_augmentations_v0( *d0[0,:,:].shape )
                    )
                if ax0==1:
                    # self.datavols.append(d0.permute(2,0,1)) #unlike tranpose, this gets a different view, so no copy done
                    # self.labelsvols.append(l0.permute(2,0,1))
                    self.datavols.append( np.rot90(d0, 1, axes=(0,2)) )
                    self.labelsvols.append(np.rot90(l0, 1, axes=(0,2)))
                    self.tfms.append(
                        get_train_augmentations_v0( *d0[:,0,:].shape )
                    )
                if ax0==2:
                    self.datavols.append( np.rot90(d0, 1, axes=(0,1)) )
                    self.labelsvols.append(np.rot90(l0, 1, axes=(0,1)))
                    self.tfms.append(
                        get_train_augmentations_v0( *d0[:,:,0].shape )
                    )
        
        self._idx_cum_sum = np.cumsum(np.array([i.shape[0] for i in self.labelsvols]))
        self._idx_cum_sum =np.concatenate(([0],self._idx_cum_sum), axis=0) #zslices
        
        self.cuda_device = cuda_device

    def __len__(self):
        return self._idx_cum_sum[-1] #last vlaue of the cumulative sum has the number of indices

    def __getitem__(self, idx):
        
        #Check which volume and z coordinate does this index correspond to
        vol_idx = np.where(np.bitwise_and(self._idx_cum_sum[0:-1]<=idx, self._idx_cum_sum[1:]>idx))[0][0]
        z_idx = idx-self._idx_cum_sum[vol_idx]
        
        #print(f"idx:{idx}, vol_idx:{vol_idx}, z_idx:{z_idx}")
        
        data = self.datavols[vol_idx][z_idx,:,:]
        labels = self.labelsvols[vol_idx][z_idx,:,:]
        
        #print(f"data shape:{data.shape}, dtype:{dtype}")
        assert data.shape == labels.shape

        # Apply transforms
        res =self.tfms[vol_idx](image=data, mask=labels)
        #transforms returns torch tensor

        data=res['image']
        labels=res['mask']
        
        data= data.to(f"cuda:{self.cuda_device}").float()
        labels=labels.to(f"cuda:{self.cuda_device}").long()

        #return a tuple data, mask
        return data, labels

class VolumeSlicerDataset(Dataset):

    def __init__(self, datavol, axis, per_slice_tfms=None):
        assert datavol.ndim==3
        assert axis==0 or axis==1 or axis==2

        self.datavol=datavol
        self.axis=axis
        self.per_slice_tfms=per_slice_tfms

    def __len__(self):
        return self.datavol.shape[self.axis]

    def __getitem__(self, idx):
        
        data_slice=None
        if self.axis==0:
            data_slice = self.datavol[idx,:,:]
        elif self.axis==1:
            data_slice = self.datavol[0,idx,:]
        elif self.axis==2:
            data_slice = self.datavol[0,:,idx]

        data_sl_np = data_slice.numpy() #Convert to numpy as data is in torch tensor and albumentations only support numpy

        # Apply transforms
        res=data_sl_np
        if self.per_slice_tfms is not None:
            res = self.per_slice_tfms(data_sl_np)

        return res

def X_parse(X):
    #return X.to('cuda')
    #return torch.unsqueeze(X.to('cuda'),dim=1).float()
    return torch.unsqueeze(X,dim=1).float()

def y_parse(y):
    #return torch.unsqueeze(y.to('cuda'),dim=1).float()
    #return y.to('cuda')
    return y.long()

def train_loop(dataloader, model, loss_fn, optimizer, scaler, scheduler, do_log=True):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X=X_parse(X)
        # Compute prediction and loss
        pred = model(X)

        #y= y_parse(y) # to cuda
        loss = loss_fn(pred, y)

        # Backpropagation
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #optimizer.step() #step done by the scheduler
        optimizer.zero_grad()

        scheduler.step()

        if do_log and batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"batch:{batch}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]. lr:{scheduler.get_last_lr()}")

def test_loop(dataloader, model, loss_fn, metric_fn=None):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    #size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_losses=[]
    metrics=[]

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X=X_parse(X)
            #y=y_parse(y)
            pred = model(X)
            test_loss = loss_fn(pred, y).item()
            test_losses.append(test_loss)

            if not metric_fn is None:
                pred_argmax = torch.argmax(pred, dim=1)
                metric = metric_fn(pred_argmax, y).item()
                metrics.append(metric)
            # #metric
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = np.mean(np.array(test_loss))
    logging.info(f"Avg loss: {avg_loss:>8f}")

    avg_metric=None
    if not metric_fn is None:
        avg_metric = np.mean(np.array(metrics))
        logging.info(f"Avg metric: {avg_metric:>8f}")

    return {"avg_loss":avg_loss, "avg_metric":avg_metric}

def train_model(model0, dl_train, dl_test, loss_fn, optimizer, scaler, scheduler, epochs, metric_fn=None):
    epoch_test_losses=[]
    for t in range(epochs):
        logging.info(f"---- Epoch {t+1}/{epochs} ----")
        train_loop(dl_train, model0, loss_fn, optimizer, scaler, scheduler)
        test_res= test_loop(dl_test, model0, loss_fn, metric_fn=metric_fn)
        epoch_test_losses.append(test_res["avg_loss"])
    logging.info(f"Done! Final loss is : {test_res['avg_loss']}, and metric is: {test_res['avg_metric']}")

def get_train_augmentations_v0(h,w):

    def get_nearest_multiple_of_32(v):
        i32 = v//32
        return i32*32

    img_h, img_w = h,w

    img_h32, img_w32 = get_nearest_multiple_of_32(img_h),  get_nearest_multiple_of_32(img_w)
    assert img_h32>0 and img_w>0

    tfms0 =alb.Compose(
                [
                alb.RandomSizedCrop(
                    min_max_height= (img_h32//2, img_h32),
                    height=img_h32,
                    width=img_w32 ,
                    p=0.5,
                ),
                #Deciding what resizing augmentations is difficult not kowing what
                # sizes the images can be different

                alb.VerticalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.Transpose(p=0.5),
                alb.OneOf(
                    [
                        alb.ElasticTransform(
                            alpha=120, sigma=120 * 0.07, alpha_affine=120 * 0.04, p=0.5
                        ),
                        alb.GridDistortion(p=0.5),
                        alb.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                    ],
                    p=0.5,
                ),
                alb.CLAHE(p=0.5),
                alb.OneOf([alb.RandomBrightnessContrast(p=0.5),alb.RandomGamma(p=0.5)], p=0.5),
                alb.pytorch.ToTensorV2()
                ]
            )
    return tfms0