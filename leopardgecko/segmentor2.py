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


Major change.
Rather than using a class, I will use global functions and variables

Most of the stuff here is from /scripts/developing_segmentor2.ipynb

"""

import numpy as np
#import dask.array as da
import tempfile
from pathlib import Path

import tempfile
import logging
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import albumentations as alb
import albumentations.pytorch

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

import matplotlib.pyplot as plt

import logging
#logging.basicConfig(level=logging.INFO)

#import tifffile
import h5py
from tqdm import tqdm
import random
import gc


#Default settings

data_vol_norm_process_str = "mean_stdev_3" #standard clipping

nn1_loss_criterion='DiceLoss'
nn1_eval_metric='MeanIoU'
nn1_lr=1e-5
nn1_max_lr=3e-2

nn1_train_epochs = 10
# nn1_train_epochs = 5 # debug

nn1_batch_size = 2
nn1_num_workers = 1

#Default
# nn1_models_class_generator = [{
# 'class':'smp', #smp: segmentation models pytorch
# 'arch': 'U_Net',
# 'encoder_name': 'resnet34',
# 'encoder_weights': 'imagenet', # TODO: support for using existing models (loading)
# 'in_nchannels':1,
# 'nclasses':3,
# }]

nn1_axes_to_models_indices = [0,1,2] # By default use the same model for all axes
# To use different models, use [0,1,2] for model0 along z, model1 along y, and model2 along x

temp_data_outdir = None


torch_device_str="cpu"
if torch.cuda.is_available():
    torch_device_str = "cuda:0"

# # Number of output classes. Also the max value of the training labels +1
# # Make sure you change for your data
# nclasses =3

# 
nn1_dict_gen_default = {
    'class':'smp', #smp: segmentation models pytorch
    'arch': 'U_Net',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet', # TODO: support for using existing models (loading)
    'in_nchannels':1, #greyscale
    'nclasses':3,
}

nn1_models_class_generator_default = [nn1_dict_gen_default,
    nn1_dict_gen_default.copy(),
    nn1_dict_gen_default.copy()
]

nn1_models_class_generator=None

def create_nn1_ptmodel_from_class_generator(nn1_cls_gen_dict: dict):
    """
    get NN1 segm model from dictionary item
    """
    #global torch_device_str

    logging.info("create_nn1_ptmodel_from_class_generator()")

    model0=None

    if nn1_cls_gen_dict['class'].lower()=='smp': #unet, AttentionNet (manet) and fpn
        #Segmentation models pytorch
        arch = nn1_cls_gen_dict['arch'].lower()
        if arch=="unet" or arch=="u_net":
            model_arch = smp.Unet
        elif arch=="manet":
            model_arch = smp.MAnet
        elif arch=="fpn":
            model_arch = smp.FPN
        elif "pan" in m:
            model_arch=smp.PAN
        elif "pspnet" in m:
            model_arch=smp.PSPNet
        else:
            raise ValueError(f"arch:{arch} not valid.")
        
        model0 = model_arch(
            encoder_name = nn1_cls_gen_dict['encoder_name'],
            encoder_weights = nn1_cls_gen_dict['encoder_weights'],
            in_channels = nn1_cls_gen_dict['in_nchannels'],
            classes = nn1_cls_gen_dict['nclasses'],
            #activation = "sigmoid" # Whether to use activation or not, depends whether the loss function require slogits or not
            activation = None
            )
    else:
        raise ValueError(f"class {nn1_cls_gen_dict['class']} not supported.")
    
    # TODO: add other 2D model support, not just SMPs

    #model0.to(torch_device_str)

    return model0

NN1_models = None
nn1_axes_to_models_indices = [0,1,2] #default

def update_nn1_models_from_generators():
    """
    Sets up NN1_models based in information from nn1_models_class_generator
    Also sends torch model to torch_device_str
    """
    global nn1_models_class_generator
    global NN1_models
    
    logging.info("update_NN1_models_from_generators()")
    if nn1_models_class_generator is not None:
        NN1_models=[]
        if (len(nn1_models_class_generator)==0):
            raise ValueError("No elements in nn1_models_class_generator to generate NN1 models")
        logging.info(f"{len(nn1_models_class_generator)} NN1 models to be created")
        for cg0 in nn1_models_class_generator:
            m0 = create_nn1_ptmodel_from_class_generator(cg0)
            
            m0.to(torch_device_str)
            
            NN1_models.append(m0)

        if len(NN1_models)==0:
            logging.info("No NN1 models")
            NN1_models=None



def normalise_voldata_to_stdev_3(datavol):
    # Clip data to -3*stdev and +3*stdev and
    # normalises to values between 0 and 1
    # Result is returned as float
    logging.info("normalise_voldata_to_stdev_3()")
    d0_mean = np.mean(datavol)
    d0_std = np.std(datavol)

    if d0_std==0:
        raise ValueError("Error. Stdev of data volume is zero.")
    
    d0_corr = (datavol.astype(np.float32) - d0_mean) / d0_std
    d0_corr = (np.clip(d0_corr, -3.0, 3.0) +3.0) / 6.0
    
    return (d0_corr*255).astype(np.uint8)


def get_train_augmentations_v0(h,w):
    # Gets alb augmentations based on image size height x width
    # Initial RandomSizedCrop resizes to nearest multiple of 32

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

class NN1_train_input_dataset_along_axes(Dataset):
    """
    Custom pytorch class for slicing several datavolumes (from list)
    along a one or a list of axis
    This is useful for training
    """
    def __init__(self, datavols_list, labelsvols_list, axes=[0,1,2]):
        #global torch_device_str
        logging.info(f"NN1_train_input_dataset_along_axes __init__ with len(data):{len(datavols_list)}, axes:{axes}")


        self.datavols_list = datavols_list
        self.labelsvols_list = labelsvols_list

        self.axes = axes
        if isinstance(axes, int): 
            self.axes=[axes]
        
        #Augmentations will fail if data is not uint8
        #Give warning here
        if np.any([d.dtype!=np.uint8 for d in datavols_list]):
            logging.warning("Some data is not uint8 format. Augmentations may fail.")

        #given an idx number, retrive the item, axis, slice number and transform
        self._idx_to_item=[]
        self._idx_to_ax=[]
        self._idx_to_slicen=[]
        self._idx_to_tfms = []

        #total_slices=0
        for id, d0 in enumerate(datavols_list):
            for ia, ax0 in enumerate(axes):
                nslices=d0.shape[ax0]
                #total_slices+= nslices

                id0_to_item = [id]*nslices
                self._idx_to_item.extend(id0_to_item)

                ax0_to_item = [ax0]*nslices
                self._idx_to_ax.extend(ax0_to_item)

                slice_range = np.arange(0,nslices).tolist()
                self._idx_to_slicen.extend(slice_range)

                if ax0==0:
                    t0 = get_train_augmentations_v0( *d0[0,:,:].shape )
                elif ax0==1:
                    t0 = get_train_augmentations_v0( *d0[:,0,:].shape )
                elif ax0==2:
                    t0 = get_train_augmentations_v0( *d0[:,:,0].shape )
                else:
                    raise ValueError(f"ax0 {ax0} not valid")
                self._idx_to_tfms.extend([t0]*nslices)

        total_slices = len(self._idx_to_item)

        assert total_slices==len(self._idx_to_ax) and total_slices==len(self._idx_to_ax) and total_slices==len(self._idx_to_slicen) and total_slices==len(self._idx_to_tfms)

        self.len = total_slices


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        global torch_device_str

        it = self._idx_to_item[idx]
        ax = self._idx_to_ax[idx]
        slicen = self._idx_to_slicen[idx]
        tfms = self._idx_to_tfms[idx]
        
        if ax==0:
            data_slice = self.datavols_list[it][slicen,:,:]
            labels_slice = self.labelsvols_list[it][slicen,:,:]
        elif ax==1:
            data_slice = self.datavols_list[it][:,slicen,:]
            labels_slice = self.labelsvols_list[it][:,slicen,:]
        elif ax==2:
            data_slice = self.datavols_list[it][:,:,slicen]
            labels_slice = self.labelsvols_list[it][:,:,slicen]
        else:
            raise ValueError(f"ax {ax} not valid")

        assert data_slice.shape == labels_slice.shape

        # Apply transforms
        res =tfms(image=data_slice, mask=labels_slice)

        data=res['image']
        labels=res['mask']
        
        data= data.to(torch_device_str).float()
        labels=labels.to(torch_device_str).long()

        #return a tuple data, mask
        return data, labels

# def create_nn1_dls_from_datalists(traindata_list, trainlabels_list):
#     """
#     Create tains and test dataloaders from training data and labels
#     Test and train data slices are split in proportions 0.8, 0.2
#     Returns dataloaders_train, dataloaders_test as a list

#     """

#     global nn1_batch_size

#     dataloaders_train=[]
#     dataloaders_test=[]

#     for i in range(len(NN1_models)):
#         #Gets the axes that the NN1 model is supposed to be used
#         model_axes= np.flatnonzero(
#             np.array(nn1_axes_to_models_indices) == i
#         ).tolist()

#         dl_train=None
#         dl_test=None

#         if len(model_axes)>0:

#             ds0 = NN1_train_input_dataset_along_axes(
#                 traindata_list,
#                 trainlabels_list,
#                 axes=model_axes
#             )

#             dset1, dset2 = torch.utils.data.random_split(ds0, [0.8,0.2])

#             dl_train = DataLoader(dset1, batch_size=nn1_batch_size, shuffle=True)
#             dl_test = DataLoader(dset2, batch_size=nn1_batch_size, shuffle=True)

#         dataloaders_train.append(dl_train)
#         dataloaders_test.append(dl_test)


nn1_loss_func_and_activ = None
nn1_train_CEloss_weights = None
NN1_LOSS_STRS = ["crossentropyloss", "diceloss"]
def update_nn1_loss_func_and_activ():
    """
    Updates nn1_loss_func_and_activ based in variable nn1_loss_criterion
    """
    global nn1_loss_func_and_activ, nn1_loss_criterion
    global torch_device_str

    activ = torch.nn.Sigmoid()
    if "crossentropyloss" in nn1_loss_criterion.lower():

        #nn1_loss_func = torch.nn.CrossEntropyLoss().to(torch_device_str) # expects logits!
        if nn2_train_CEloss_weights is None:
            nn1_loss_func= nn.CrossEntropyLoss().to(torch_device_str)
        else:
            weights_tc = torch.Tensor(nn2_train_CEloss_weights).to(torch_device_str)
            nn1_loss_func=nn.CrossEntropyLoss(weights_tc).to(torch_device_str)
        
        # or can use
        # nn1_loss_func = torch.nn.functional.cross_entropy(pred_logits, target) but no weights i think
        
        nn1_loss_func_and_activ= {"func":nn1_loss_func, "activ":activ}
    elif "diceloss" in nn1_loss_criterion.lower():
        nn1_loss_func = smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(torch_device_str)
        nn1_loss_func_and_activ= {"func":nn1_loss_func, "activ":None}
    else:
        raise ValueError(f"{nn1_loss_criterion} not a valid loss criteria")

nn1_metric_func = None
def update_nn1_metric_func():
    """
    Updates nn1_metric_func based in variable nn1_eval_metric
    """
    global nn1_metric_func, nn1_eval_metric
    global torch_device_str

    if "iou" in nn1_eval_metric.lower():
        nn1_metric_func = segmentation_models_pytorch.utils.metrics.IoU()
    elif "dice" in nn1_eval_metric.lower() or "fscore" in nn1_eval_metric.lower():
        nn1_metric_func = segmentation_models_pytorch.utils.metrics.Fscore()
    elif "accuracy" in nn1_eval_metric.lower():
        nn1_metric_func = segmentation_models_pytorch.utils.metrics.Accuracy()



def train_loop(dataloader, model, loss_func_and_activ, optimizer, scaler, scheduler, do_log=True):
    """
    Generic function for training a pytorch model
    with provided model, loss_func_and_activ, optimizer, scaler, scheduler

    This trains for a single epoch only.

    Returns nothing, but the model weigths have certainly changed for better.
    """
    loss_fn = loss_func_and_activ["func"]
    activ_fn = loss_func_and_activ["activ"]
    
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #X=X_parse(X)
        # Compute prediction and loss
        pred = model(X)

        if activ_fn is not None:
            pred = activ_fn(pred)

        loss = loss_fn(pred, y)

        # Backpropagation
        #loss.backward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #optimizer.step() #step done by the scheduler
        optimizer.zero_grad()

        scheduler.step()

        if do_log and batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logging.info(f"batch:{batch}  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]. lr:{scheduler.get_last_lr()}")

def test_loop(dataloader, model, loss_func_and_activ, metric_fn=None):
    """
    Generic function for runing test loop with provided parameters
    Gets the average loss and average metric calcualted on the test data.
    It can also calculate metric value

    Note that the metric is calculated per batch and then averaged on all batches.
    This may not be the most a appropriate way to calculate the metrics.

    """
    
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    #size = len(dataloader.dataset)
    #num_batches = len(dataloader)

    loss_fn = loss_func_and_activ["func"]
    activ_fn = loss_func_and_activ["activ"]

    test_losses=[]
    test_metrics=[]

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            #X=X_parse(X)
            #y=y_parse(y)
            pred = model(X)

            if activ_fn is not None:
                pred = activ_fn(pred)

            loss = loss_fn(pred, y)

            test_loss = loss.item()
            test_losses.append(test_loss)
            
            if metric_fn is not None:
                pred_argmax = torch.argmax(pred, dim=1)
                metric = metric_fn(pred_argmax, y).item()
                test_metrics.append(metric)
            # #metric
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = np.mean(np.array(test_losses))
    logging.info(f"Avg loss: {avg_loss:>8f}")

    avg_metric=None
    if not metric_fn is None:
        avg_metric = np.mean(np.array(test_metrics))
        logging.info(f"Avg metric: {avg_metric:>8f}")

    return {"avg_loss":avg_loss, "avg_metric":avg_metric, "test_metrics":test_metrics, "test_losses":test_losses}

def train_model(model0, dl_train, dl_test, loss_func_and_activ, optimizer, scaler, scheduler, epochs, metric_fn=None):
    """
    Trains and tests model, for a number of epochs
    """

    logging.info("train_model()")
    test_results=[]
    for t in range(epochs):
        logging.info(f"---- Epoch {t+1}/{epochs} ----")
        train_loop(dl_train, model0, loss_func_and_activ, optimizer, scaler, scheduler)

        test_res=None
        if dl_test is not None:
            test_res= test_loop(dl_test, model0, loss_func_and_activ, metric_fn=metric_fn)
            test_results.append(test_res)
    logging.info(f"Done!")
    if dl_test is not None:
        logging.info(f"Final test loss is : {test_res['avg_loss']}, and metric is: {test_res['avg_metric']}")
    return {"test_results": test_results}


#Utility function to save nn1 prediction data
def _save_pred_data(folder, data, count,axis, rot):
    # Saves predicted data to h5 file in tempdir and return file path in case it is needed
    file_path = f"{folder}/pred_{count}_{axis}_{rot}.h5"

    logging.info(f"Saving data of shape {data.shape} to {file_path}.")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("/data", data=data)

    return file_path

last_train_nn1_progress=None
def train_nn1(traindata_list, trainlabels_list):
    """
    Train each of the NN1 models individually taking into
    consideration the axis they are associated to
    
    """
    global NN1_models
    global nn1_axes_to_models_indices
    global nn1_batch_size
    global last_train_nn1_progress
    global nn1_train_epochs

    if NN1_models is None:
        raise ValueError("No NN1 models to train")
    
    #reverse nn1_axes_to_models_indices to get model to axis
    def _reverse(n0):
        nmodel_max = max(n0)
        models_indices = []
        for im in range(nmodel_max+1):
            model_indices = np.flatnonzero( np.array(n0)==im)
            models_indices.append(model_indices.tolist())
        return models_indices
    
    models_to_axis = _reverse(nn1_axes_to_models_indices)
    logging.info(f"models_to_axis: {models_to_axis}")

    if len(models_to_axis)==0:
        raise ValueError("models_to_axis has no elements")
    
    last_train_nn1_progress={}
    for imodel, axs in enumerate(models_to_axis):

        if len(axs)==0:
            logging.info(f"No axis for model: {imodel}. Skipping training.")
            break
        
        logging.info(f"Training imodel {imodel}, along axes {axs}.")

        logging.info("Setting up datasets and dataloaders.")
        # Create dataloaders for training
        ds0 = NN1_train_input_dataset_along_axes(
            traindata_list,
            trainlabels_list,
            axes = axs
        )

        dset1, dset2 = torch.utils.data.random_split(ds0, [0.8,0.2])

        dl_train = DataLoader(dset1, batch_size=nn1_batch_size, shuffle=True)
        dl_test = DataLoader(dset2, batch_size=nn1_batch_size, shuffle=True)

        model = NN1_models[imodel]

        # setup optimizer and scaler
        #Setup optimizer and scaler
        logging.info("Setting up optimizer and scheduler.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=nn1_lr)
        scaler=torch.cuda.amp.GradScaler()

        epochs = nn1_train_epochs #global
        #epochs = 10

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr= nn1_max_lr,
            steps_per_epoch=len(dl_train),
            epochs=epochs,
            #pct_start=0.1, #default=0.3
            )
        
        if nn1_loss_func_and_activ is None:
            update_nn1_loss_func_and_activ()
        if nn1_metric_func is None:
            update_nn1_metric_func()

        #train model here
        logging.info("Training launching.")

        train_prgr = train_model(model, dl_train, dl_test, nn1_loss_func_and_activ, optimizer, scaler, scheduler,
            epochs=epochs,
            metric_fn=nn1_metric_func
            )
        last_train_nn1_progress[imodel]=train_prgr

        logging.info(f"Training model {imodel} along {axs} complete.")



class VolumeSlicerDataset(Dataset):
    """
    Dataset class to slice a single volume along a single axis
    This is useful for running predictions
    """
    def __init__(self, datavol, axis, per_slice_tfms=None, device_str=None):
        #global torch_device_str

        assert datavol.ndim==3
        assert axis==0 or axis==1 or axis==2

        self.datavol=datavol
        self.axis=axis
        self.per_slice_tfms=per_slice_tfms
        self.device_str = device_str

    def __len__(self):
        return self.datavol.shape[self.axis]

    def __getitem__(self, idx):
        global torch_device_str

        data_slice=None
        if self.axis==0:
            data_slice = self.datavol[idx,:,:]
        elif self.axis==1:
            data_slice = self.datavol[:,idx,:]
        elif self.axis==2:
            data_slice = self.datavol[:,:,idx]

        res = data_slice
        # Apply transform
        if self.per_slice_tfms is not None:
            res = self.per_slice_tfms(data_slice)

        #Convert to tensor and send to device
        res_torch = torch.unsqueeze(torch.from_numpy(res), dim=0).float().to(torch_device_str)

        return res_torch


def predict_nn1_slices_along_axis_1(datavol, axis):
    """
    Inference of a single datavol along the given axis
    using the respective NN1_models
    """

    global NN1_models
    global nn1_axes_to_models_indices
    global torch_device_str

    ds0 = VolumeSlicerDataset(datavol, axis , per_slice_tfms=None, device_str=torch_device_str)
    dl0 = DataLoader(dataset=ds0, batch_size=nn1_batch_size, shuffle=False)

    # Get correct model
    model_index = nn1_axes_to_models_indices[axis]
    #model = NN1_models[model_index]
    model = NN1_models[model_index]
    logging.info(f"axis:{axis}, use model_index: {model_index}")

    model.eval()
    
    SM_func = torch.nn.Softmax(dim=1)

    preds_list = []
    labels_list = []
    for ibatch, x in enumerate(dl0):
        # x.shape is (batchsize, 1, 256,256) with 256 being the imagesize
        X= model(x)
        #pred shape is (batchsize, 3, 256, 256)

        pred_probs_slice = SM_func(X) #Convert to probabilities

        # get labels using argmax
        lbl_slice = torch.argmax(pred_probs_slice, dim=1)
        #labels_list.append(lbl_slice)

        # need to move away from device, otherwise it uses too much VRAM
        pred_probs_slice_np = pred_probs_slice.detach().cpu().numpy()
        lbl_slice_np = lbl_slice.detach().cpu().numpy().astype(np.uint8)

        preds_list.append(pred_probs_slice_np)
        labels_list.append(lbl_slice_np)

    logging.info("Prediction of all slices complete. Now stacking and getting the right orientation.")
    # stack slices
    preds_list_conc = np.concatenate(preds_list, axis=0) # shape will be (256,3,256,256)
    labels_pred_conc = np.concatenate(labels_list, axis=0)

    pred_oriented = None
    labels_oriented = None
    if axis==0:
        pred_oriented = np.transpose(preds_list_conc, axes=(1,0,2,3))
        labels_oriented = labels_pred_conc # no need to orient
    elif axis==1:
        pred_oriented = np.transpose(preds_list_conc, axes=(1,2,0,3))
        labels_oriented = np.transpose(labels_pred_conc, axes=(1,0,2))
    elif axis==2:
        pred_oriented = np.transpose(preds_list_conc, axes=(1,2,3,0))
        labels_oriented = np.transpose(labels_pred_conc, axes=(1,2,0))

    #with pred_oriented note that class probability is at the start
    return pred_oriented, labels_oriented

last_nn1_prediction_df = None
def predict_nn1(data_to_predict_l, path_out_results):

    """
    Runs predictions from a list of datavolumes, by 12-way (4 rotations * 3 axis)

    It assumes that volumes have all been normalised

    Returns: a pandas dataframe listing all the files that have been generated
    with the following columns
        'pred_data_probs_filenames'
        'pred_data_labels_filenames'
        'pred_sets'
        'pred_planes'
        'pred_rots'
        'pred_ipred'
        'pred_shapes'
    
    As for predictions, for each datavol, and rotations, it predicts two datavolumes
        - probabilities for each class at each voxel
        - labels (argmax) at each voxel "labels"

    Note that prediction axis is specified not by index number but by plane
    eg: along axis Z will be specified as YX

    """
    global last_nn1_prediction_df

    logging.info("predict_NN1()")
    pred_data_probs_filenames=[] #Will store results in files, and keep the filenames as reference
    pred_data_labels_filenames=[]
    pred_sets=[]
    pred_planes=[]
    pred_rots=[]
    pred_ipred=[]
    pred_shapes=[]


    for iset, data_to_predict in enumerate(data_to_predict_l):
        logging.info(f"Data to predict iset:{iset}")
        #data_vol = np.array(data_to_predict0) #Copies

        ipred=0
        for krot in range(0, 4): #Around axis rotations
            rot_angle_degrees = krot * 90
            logging.info(f"Volume to be rotated by {rot_angle_degrees} degrees")

            #Predict 3 axis
            #YX, along Z
            # planeYX=(1,2)
            logging.info("Predicting YX slices, along Z")
            data_vol = np.array(np.rot90(data_to_predict,krot, axes=(1,2))) #rotate

            #prob0,lab0 = nn1_predict_slices_along_axis(data_vol, axis=0, device_str=cuda_str)
            prob0,lab0 = predict_nn1_slices_along_axis_1(data_vol, 0)

            #invert rotations before saving
            pred_probs = np.rot90(prob0, -krot, axes=(2,3)) 
            pred_labels = np.rot90(lab0, -krot, axes=(1,2)) #note that class is at start

            fn = _save_pred_data(path_out_results,pred_probs, iset, "YX", rot_angle_degrees)
            pred_data_probs_filenames.append(fn)
            fn = _save_pred_data(path_out_results,pred_labels, iset, "YX_labels", rot_angle_degrees)
            pred_data_labels_filenames.append(fn)
            
            pred_sets.append(iset)
            pred_planes.append("YX")
            pred_rots.append(rot_angle_degrees)
            pred_ipred.append(ipred)
            pred_shapes.append(pred_labels.shape)
            ipred+=1



            #ZX
            logging.info("Predicting ZX slices, along Y")
            #planeZX=(0,2)
            data_vol = np.array(np.rot90(data_to_predict,krot, axes=(0,2))) #rotate
            #prob0,lab0 = nn1_predict_slices_along_axis(data_vol, axis=1, device_str=cuda_str)
            prob0,lab0 = predict_nn1_slices_along_axis_1(data_vol, 1)


            pred_probs = np.rot90(prob0, -krot, axes=(1,3)) #invert rotation before saving
            pred_labels = np.rot90(lab0, -krot, axes=(0,2))

            fn = _save_pred_data(path_out_results,pred_probs, iset, "ZX", rot_angle_degrees)
            pred_data_probs_filenames.append(fn)
            fn = _save_pred_data(path_out_results,pred_labels, iset, "ZX_labels", rot_angle_degrees)
            pred_data_labels_filenames.append(fn)
            
            pred_sets.append(iset)
            pred_planes.append("ZX")
            pred_rots.append(rot_angle_degrees)
            pred_ipred.append(ipred)
            pred_shapes.append(pred_labels.shape)
            ipred+=1



            #ZY
            logging.info("Predicting ZY slices, along X")
            #planeZY=(0,1)
            data_vol = np.array(np.rot90(data_to_predict,krot, axes=(0,1))) #rotate
            #prob0,lab0 = nn1_predict_slices_along_axis(data_vol, axis=2, device_str=cuda_str)
            prob0,lab0 = predict_nn1_slices_along_axis_1(data_vol, 2)

            pred_probs = np.rot90(prob0, -krot, axes=(1,2)) #invert rotation before saving
            pred_labels = np.rot90(lab0, -krot, axes=(0,1))
            
            fn = _save_pred_data(path_out_results,pred_probs, iset, "ZY", rot_angle_degrees)
            pred_data_probs_filenames.append(fn)
            fn = _save_pred_data(path_out_results,pred_labels, iset, "ZY_labels", rot_angle_degrees)
            pred_data_labels_filenames.append(fn)
            
            pred_sets.append(iset)
            pred_planes.append("ZY")
            pred_rots.append(rot_angle_degrees)
            pred_ipred.append(ipred)
            pred_shapes.append(pred_labels.shape)
            ipred+=1

    all_pred_pd = pd.DataFrame({
        'pred_data_probs_filenames': pred_data_probs_filenames,
        'pred_data_labels_filenames': pred_data_labels_filenames,
        'pred_sets':pred_sets,
        'pred_planes':pred_planes,
        'pred_rots':pred_rots,
        'pred_ipred':pred_ipred,
        'pred_shapes': pred_shapes,
    })

    last_nn1_prediction_df = all_pred_pd
    
    return all_pred_pd


# NN2 (MLP)
nn2_model_fusion=None
nn2_MLP_model_class_generator=None
torch_device_str_nn2=torch_device_str # User will have to specify if different

nn2_MLP_model_class_generator_default = {
    "nn2_hidden_layer_sizes" : "10,10",
    "nn2_activation": 'tanh',
    "nn2_out_nclasses": 3,
    "nn2_in_nchannels": 3*12
}

class MLPClassifier(nn.Module):
    # MLP classifier with sigmoid activation

    # Should I add softmax?
    def __init__(self, input_size:int, hiden_sizes_list:list, output_size:int, activ_str:str):
        super().__init__()

        size0= input_size

        self.hidden = nn.ModuleList()

        for hls in hiden_sizes_list:
            hid_layer0 =  nn.Linear(size0, hls)
            self.hidden.append(hid_layer0)
            size0=hls
        #last layer
        self.hidden.append(nn.Linear(size0, output_size))

        if "tanh" in activ_str.lower():
            self.activ = nn.functional.tanh
        elif "relu" in activ_str.lower():
            self.activ = nn.functional.relu
        elif "sigm" in activ_str.lower():
            self.activ = nn.functional.sigmoid
        else:
            raise ValueError(f"activ_str {activ_str} not valid")

    def forward(self, x):
        # for i,hlayer in self.hidden:
        #     x= self.activ(hlayer(x))
        for i in range(len(self.hidden)-1):
            x= self.activ(self.hidden[i](x))
        
        #Last layer
        x = self.hidden[-1](x)
        
        #x = self.sigm(x)
        return x #returns logits
    
    # def predict_class_as_cpu_np(self,x):
    #     p0 = self.forward(x)
    #     pred = torch.squeeze(torch.argmax(p0, dim=1))
    #     return pred.detach().cpu().numpy()


def create_nn2_ptmodel_from_class_generator(nn2_cls_gen_dict: dict ):
    logging.info("create_nn2_ptmodel_from_class_generator()")

    hid_layers = nn2_cls_gen_dict['nn2_hidden_layer_sizes'].split(",")

    if len(hid_layers)==0:
        ValueError(f"Invalid nn2_hidden_layer_sizes : {nn2_cls_gen_dict['nn2_hidden_layer_sizes']}")

    hid_layers_num_list = list(map(int, hid_layers))
    logging.info(f"hid_layers_num_list: {hid_layers_num_list}")
    
    model0 = MLPClassifier(
        nn2_cls_gen_dict['nn2_in_nchannels'],
        hid_layers_num_list,
        nn2_cls_gen_dict['nn2_out_nclasses'],
        nn2_cls_gen_dict["nn2_activation"]
        )
    
    if "NN2_model_dict" in nn2_cls_gen_dict.keys():
        logging.info("NN2: load weights from dict")
        model0.load_state_dict(nn2_cls_gen_dict["NN2_model_dict"])
        
    return model0

def update_nn2_model_from_generator():
    """
    Sets up NN2_model_fusion based in information from ...
    Also sends torch model to torch_device_str_nn2
    """

    global nn2_MLP_model_class_generator
    global nn2_model_fusion
    global torch_device_str_nn2

    logging.info("update_NN2_model_from_generator()")

    nn2_model_fusion=None
    if nn2_MLP_model_class_generator is not None:
        nn2_model_fusion = create_nn2_ptmodel_from_class_generator(nn2_MLP_model_class_generator)
        nn2_model_fusion.to(torch_device_str_nn2)

def aggregate_data_from_pd(all_pred_df):
    """
    Aggregates data from files described in dataframe all_pred_pd
    It must have the folowing columns
    pred_data_probs_filenames, pred_sets, pred_ipred, pred_sets

    Returns a an array with th following shape
    [ iset, ipred (from 0 to 12) , probs , Z ,Y ,X ]

    """

    data_all_np6d=None

    logging.debug("Aggregating multiple sets onto a single volume data_all_np6d")
    # aggregate multiple sets for data
    dtypes_try = [ "original", "float16"]

    for i,prow in all_pred_df.iterrows():

        prob_filename = prow['pred_data_probs_filenames']
        with h5py.File(prob_filename,'r') as f:
            data0 = np.array(f["data"])

        if i==0:
            #initialise
            logging.info(f"filename:{prob_filename} , shape:{data0.shape}")
            all_shape0 = (
                all_pred_df["pred_sets"].max()+1, # needs to be adjusted
                all_pred_df["pred_ipred"].max()+1, # needs to be adjusted, perhaps can be collected from dataframe
                *data0.shape
                )

            success=False
            for dtype_str in dtypes_try:
                
                dtype0 = data0.dtype
                if dtype_str=="float16":
                    dtype0=np.float16

                logging.info(f"dtype_str:{dtype_str}, dtype0:{dtype0}")

                try:
                    data_all_np6d=np.zeros( all_shape0 , dtype=dtype0) # Can lead to RAM errors
                    success=True
                except Exception as e:
                    logging.error(f"Setting up data container using dtype {dtype0} failed with error:{str(e)}.")
                    
            if not success:
                raise RuntimeError("Could not create array to aggregate data.")

        ipred=prow['pred_ipred']
        iset=prow['pred_sets']

        if dtype_str=="original":
            data_all_np6d[iset,ipred, :,:,:, :] = data0
        else:
            data_all_np6d[iset,ipred, :,:,:, :] = data0.astype(np.float16)
        break


    return data_all_np6d

def aggregate_data_from_pd_iset(all_pred_df, iset=0):
    """
    Aggregates data from files described in dataframe all_pred_pd
    Only for the iset value provided (default=0)

    all_pred_df dataframe must have the folowing columns
    pred_data_probs_filenames, pred_sets, pred_ipred, pred_sets

    Returns a an array with th following shape (5D)
    [ ipred (from 0 to 12) , probs , Z ,Y ,X ]

    Data read from pd df is converted immediately to float16 to save RAM
    """

    data_all_np5d=None

    logging.info(f"aggregate_data_from_pd_iset() with iset:{iset}")

    #res_pd.loc[res_pd["pred_sets"]==0]
    # gets only the rows with matching iset value
    df_iset = all_pred_df.loc[all_pred_df["pred_sets"]==iset]

    for i,prow in df_iset.iterrows(): # Note: i may not start at 0

        prob_filename = prow['pred_data_probs_filenames']
        
        #Load data
        with h5py.File(prob_filename,'r') as f:
            #data0 = np.array(f["data"])
            data0 = np.array(f["data"]).astype(np.float16) #convert here

        #if i==0:
        if data_all_np5d is None:
            #initialise
            logging.info(f"Setting up based on first file filename:{prob_filename} , shape:{data0.shape}")
            all_shape0 = (
                all_pred_df["pred_ipred"].max()+1, # needs to be adjusted, perhaps can be collected from dataframe
                *data0.shape
                )

            # dtype0 = data0.dtype
            # logging.info(f"dtype0:{dtype0}")

            #data_all_np5d=np.zeros( all_shape0 , dtype=dtype0) # Can lead to RAM errors
            data_all_np5d=np.zeros( all_shape0 , dtype=np.float16)

        ipred=prow['pred_ipred']
        #iset=prow['pred_sets']

        data_all_np5d[ipred, :,:,:, :] = data0

    return data_all_np5d




#nn2_max_iter = 1000
#nn2_ntrain = 262144 #Number of random voxels to be considered for dataset training nn2
#nn2_ntrain = 2**17
nn2_ntrain = 2**18

nn2_train_epochs = 10
# nn2_train_epochs = 10 #debug
nn2_batch_size = 4096
nn2_lr = 1e-6
nn2_max_lr = 5e-2
#nn2_loss_func_and_activ=None

last_train_nn2_progress = None

nn2_train_do_class_balance= False
nn2_ntrain_in_class_balance = 2**16

nn2_train_CEloss_weights = None # Weights for the cross entropy loss function as a list

def train_nn2(data_all_np6d, trainlabels_list):

    if nn2_train_do_class_balance:
        return train_nn2_class_balanced(data_all_np6d, trainlabels_list)

    return train_nn2_default(data_all_np6d, trainlabels_list)


def _train_nn2_with_DLs(nn2_train_loader, nn2_test_loader):

    global nn2_train_epochs
    global nn2_model_fusion
    global nn2_batch_size
    global nn2_lr
    global nn2_max_lr
    #global nn2_loss_func_and_activ
    global torch_device_str_nn2
    global last_train_nn2_progress

    model=nn2_model_fusion
    model.to(torch_device_str_nn2)# ensure is in the correct device

    optimizer = torch.optim.AdamW(model.parameters(), lr=nn2_lr)
    scaler=torch.cuda.amp.GradScaler() # Does not work with CPU tensors!!

    epochs = nn2_train_epochs
    #epochs = 10

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr= nn2_max_lr,
        steps_per_epoch=len(nn2_train_loader),
        epochs=epochs,
        #pct_start=0.1, #default=0.3
        )
    if nn2_train_CEloss_weights is None:
        nn2_loss_func_and_activ= {"func": nn.CrossEntropyLoss().to(torch_device_str_nn2), "activ":None}
    else:
        weights_tc = torch.Tensor(nn2_train_CEloss_weights).to(torch_device_str_nn2) #convert to tensor
        nn2_loss_func_and_activ= {"func": nn.CrossEntropyLoss(weights_tc).to(torch_device_str_nn2), "activ":None}

    logging.info("Beggining training NN2.")

    last_train_nn2_progress = train_model(
        model,
        nn2_train_loader,
        nn2_test_loader, # use train data as test?
        nn2_loss_func_and_activ,
        optimizer, scaler, scheduler,
        epochs=epochs,
        metric_fn = segmentation_models_pytorch.utils.metrics.Accuracy()
    )

    logging.info("Training NN2 complete.")


def train_nn2_default(data_all_np6d, trainlabels_list):
    """
    data_all_np6d: per voxel per class probabilites of predictions.
    This can be collected using aggregate_data_from_pd() with output from predict_nn1()

    Typical shape from nsets (=1 if only one training volume) prediction datavolumes
    with shape 256x256x256, 3 class, 12 predictions,
    (nsets, 12, 3, 256, 256, 256)

    and corresponding labels as a list with a single int volume with shape (256,256,256)
    or np.array with shape (nsets,256,256,256)

    """
    # TODO: This training is not great for labels that are sparse.
    # Try to balance classes

    global nn2_model_fusion
    global nn2_ntrain
    global torch_device_str_nn2

    logging.info(f"NN2_train()")
    logging.info(f"data_all_np5d.shape:{data_all_np6d.shape}, len(trainlabels_list): {len(trainlabels_list)}")

    if nn2_model_fusion is None:
        raise ValueError("No NN2_model_fusion setup. Please make sure you created by either using update_NN2_model_from_generator() or by loading")

    #test dataset is a quarter of train dataset
    ntest = int(nn2_ntrain//4)

    data_ordered = np.transpose( data_all_np6d , axes=(0,3,4,5,1,2)) # turn to [ iset, Z ,Y ,X, ipred (from 0 to 12) , probs]

    # This below can cause out of RAM, because it needs to allocate new RAM for the data
    # according to the np.reshape help, it will create a new view IF POSSIBLE.
    # Otherwise it will copy the data
    # TODO: Create a routine that does not need to copy the whole volume
    # data_flat_for_mlp = p0.reshape( (np.prod(p0.shape[:4]), p0.shape[4]*p0.shape[5])) # shapesizes [ nset*nZ*nY*nX, npreds*nclasses] with typically npreds=12
    # logging.info(f"data_flat_for_mlp.shape: {data_flat_for_mlp.shape}")

    trainlabels_list_np = np.array(trainlabels_list)
    # label_flat_for_mlp = trainlabels_list_np.ravel()
    # logging.info(f"label_flat_for_mlp.shape: {label_flat_for_mlp.shape}")

    # X_train= torch.from_numpy(data_flat_for_mlp).float()
    # y_train= torch.from_numpy(label_flat_for_mlp).long()

    logging.info("Selecting only nn2_ntrain voxel coordinates from data and ground truth for training")
    
    #rand_indices = torch.randperm(X_train.shape[0])
    #subset_indices = rand_indices[:nn2_ntrain]

    # X_train_subset = X_train[subset_indices,:].to(torch_device_str_nn2)
    # y_train_subset = y_train[subset_indices].to(torch_device_str_nn2)

    # logging.info("Using _generate_unique_random_numbers() to generate unique indices")
    def _generate_unique_random_numbers(num_numbers, min_value, max_value):
        if num_numbers > max_value - min_value + 1:
            raise ValueError("Cannot generate more unique numbers than the range allows.")

        unique_numbers = set()
        while len(unique_numbers) < num_numbers:
            random_number = random.randint(min_value, max_value)
            unique_numbers.add(random_number)

        return list(unique_numbers)
    
    # rand_indices = np.array(_generate_unique_random_numbers(nn2_ntrain+ntest,0,data_flat_for_mlp.shape[0]-1))

    # assert len(rand_indices) == nn2_ntrain+ntest

    # subset_indices=rand_indices[:nn2_ntrain]

    # X_train_subset = torch.from_numpy(data_flat_for_mlp[subset_indices,:].astype(np.float32)).to(torch_device_str_nn2)
    # y_train_subset = torch.from_numpy(label_flat_for_mlp[subset_indices].astype(np.int16)).long().to(torch_device_str_nn2)

    # Create X_train_subset and y_train_subset as a torch (.to(torch_device_str_nn2)) object
    # make sure it is in the format [ nelements, 12*nclasses]

    nelements = np.prod(data_ordered.shape[:4])
    ninputs = data_ordered.shape[4]*data_ordered.shape[5]

    rand_indices = _generate_unique_random_numbers(nn2_ntrain+ntest, 0, nelements-1)

    #logging.info(f"rand_indices: {rand_indices}")

    #initialise
    X_train_test_subset = np.zeros( (nn2_ntrain+ntest, ninputs), dtype=np.float32)
    y_train_test_subset = np.zeros( (nn2_ntrain+ntest) , dtype=np.int16)

    idx_set_Z_Y_X = np.unravel_index( rand_indices, shape = data_ordered.shape[:4] )
    idx_set_Z_Y_X_t = np.transpose(np.array(idx_set_Z_Y_X))
    
    #print(f"idx_set_Z_Y_X_t:{idx_set_Z_Y_X_t}")

    #idx_set_Z_Y_X_train = idx_set_Z_Y_X_t[:nn2_ntrain]
    for i, idx0 in enumerate(idx_set_Z_Y_X_t):
        #print(f"i:{i}, idx0:{idx0}") #debug
        inp_X = data_ordered[*idx0,:,:].ravel()
        X_train_test_subset[i,:] = inp_X

        inp_y = trainlabels_list_np[*idx0] #error
        y_train_test_subset[i] = inp_y

    X_train_subset_t = torch.from_numpy(X_train_test_subset[:nn2_ntrain]).to(torch_device_str_nn2)
    y_train_subset_t = torch.from_numpy(y_train_test_subset[:nn2_ntrain]).long().to(torch_device_str_nn2)

    logging.info("X_train_subset_t and y_train_subset_t created")

    dataset_X_y_train = TensorDataset(X_train_subset_t, y_train_subset_t)
    nn2_train_loader = DataLoader(dataset_X_y_train, batch_size=nn2_batch_size, shuffle=True)

    logging.info("dataset_X_y_train created")


    # test datasets
    logging.info("Creating test dataset")
    
    X_test_subset_t = torch.from_numpy(X_train_test_subset[nn2_ntrain:nn2_ntrain+ntest]).to(torch_device_str_nn2)
    y_test_subset_t = torch.from_numpy(y_train_test_subset[nn2_ntrain:nn2_ntrain+ntest]).long().to(torch_device_str_nn2)

    logging.info("X_test_subset_t and y_test_subset_t created")

    dataset_X_y_test = TensorDataset(X_test_subset_t, y_test_subset_t)
    nn2_test_loader = DataLoader(dataset_X_y_test, batch_size=nn2_batch_size, shuffle=True)

    logging.info("dataset_X_y_test created")

    _train_nn2_with_DLs(nn2_train_loader, nn2_test_loader)


def train_nn2_class_balanced(data_all_np6d, trainlabels_list):
    """
    data_all_np6d: per voxel per class probabilites of predictions.
    This can be collected using aggregate_data_from_pd() with output from predict_nn1()

    Typical shape from nsets (=1 if only one training volume) prediction datavolumes
    with shape 256x256x256, 3 class, 12 predictions,
    (nsets, 12, 3, 256, 256, 256)

    and corresponding labels as a list with a single int volume with shape (256,256,256)
    or np.array with shape (nsets,256,256,256)

    """

    global nn2_model_fusion
    #global nn2_ntrain
    global torch_device_str_nn2
    global nn2_ntrain_in_class_balance

    logging.info(f"train_nn2_class_balanced()")
    logging.info(f"data_all_np5d.shape:{data_all_np6d.shape}, len(trainlabels_list): {len(trainlabels_list)}")
    logging.info(f"nn2_ntrain_in_class_balance:{nn2_ntrain_in_class_balance}")#

    if nn2_model_fusion is None:
        raise ValueError("No NN2_model_fusion setup. Please make sure you created by either using update_NN2_model_from_generator() or by loading")

    data_ordered = np.transpose( data_all_np6d , axes=(0,3,4,5,1,2)) # turn to [ iset, Z ,Y ,X, ipred (from 0 to 12) , probs]

    trainlabels_list_np = np.array(trainlabels_list)

    nclasses = np.max(trainlabels_list_np)+1
    nvoxels_per_class = [ np.count_nonzero( trainlabels_list_np==i ) for i in range(nclasses) ]
    logging.info(f"nclasses estimated from max: {nclasses}, nvoxels_per_class:{nvoxels_per_class}")


    logging.info("Adjusting number of elements.")


    nfrac = nn2_ntrain_in_class_balance // (4*nclasses)

    max_items_per_class = 5*nfrac
    
    thrs_vox_per_class = np.array(nvoxels_per_class)//16

    if np.any( thrs_vox_per_class < max_items_per_class):
        logging.info(f"Some thrs_vox_per_class {thrs_vox_per_class} are smaller than max_items_per_class {max_items_per_class}")
        nfrac = np.min(thrs_vox_per_class)//5
        max_items_per_class= 5*nfrac
        logging.info(f"Adjusting max_items_per_class to {max_items_per_class}")
    
    #Re-Adjusts
    ntrain = nfrac*4*nclasses
    ntest = nfrac*nclasses #test dataset is a quarter of train dataset
    ntotal = ntrain+ntest

    logging.info(f"max_items_per_class: {max_items_per_class}, ntotal (adjusted):{ntotal}")


    logging.info("Selecting only nn2_ntrain_in_class_balance voxel coordinates from data and ground truth for training by balancing class labels.")

    nvoxels = np.prod(data_ordered.shape[:4])
    ninputs = data_ordered.shape[4]*data_ordered.shape[5]

    #initialise
    X_train_test_subset = np.zeros( (ntotal, ninputs), dtype=np.float32)
    y_train_test_subset = np.zeros( (ntotal) , dtype=np.int16)

    # Collect datapoints

    #while True: #DANGER
    count_per_class = np.zeros((nclasses), dtype=np.int64)
    unique_flat_idxs = []
    shape0 = data_ordered.shape[:4]

    ielement = 0 #counter
    with tqdm(total=ntotal) as tqdm_pbar:
        for i in range(nvoxels): # *10 to impose a timeout, normally it should simply exit with a break
            
            random_flat_idx = random.randint(0, nvoxels-1)

            if random_flat_idx not in unique_flat_idxs:
                #Apply unravel to a single element
                coord = np.transpose(np.unravel_index( [random_flat_idx], shape0 ))[0]

                # get data point and label
                inp_X = data_ordered[*coord,:,:].ravel()
                inp_y = trainlabels_list_np[*coord]

                class_i = int(inp_y)

                if count_per_class[class_i] < max_items_per_class:
                    #can add this element
                    count_per_class[class_i]+=1
                    X_train_test_subset[ielement]=inp_X
                    y_train_test_subset[ielement]=inp_y

                    # if ielement % 4096 == 100:
                    #     logging.info(f"ielement: {ielement}, inp_y:{inp_y}, count_per_class:{count_per_class}")

                    ielement+=1
                    tqdm_pbar.update(1) # update increases by the value speicified

                    unique_flat_idxs.append(random_flat_idx)
                
                if ielement>= ntotal:
                    logging.info(f"Reached ielement>= ntotal : {ielement}>={ntotal}. Exiting for loop")
                    break

        else:
            #Reach the end of the loop, number of elements should be adusted or just throw error
            raise OverflowError(f"Reached the end of loop without collecting enough data points. count_per_class:{count_per_class}")

    assert len(unique_flat_idxs)==ntotal

    logging.info(f"count_per_class:{count_per_class}")

    X_train_subset_t = torch.from_numpy(X_train_test_subset[:ntrain]).to(torch_device_str_nn2)
    y_train_subset_t = torch.from_numpy(y_train_test_subset[:ntrain]).long().to(torch_device_str_nn2)

    logging.info("X_train_subset_t and y_train_subset_t created")

    dataset_X_y_train = TensorDataset(X_train_subset_t, y_train_subset_t)
    nn2_train_loader = DataLoader(dataset_X_y_train, batch_size=nn2_batch_size, shuffle=True)

    logging.info("dataset_X_y_train created")

    # test datasets
    logging.info("Creating test dataset")
    
    X_test_subset_t = torch.from_numpy(X_train_test_subset[ntrain:ntrain+ntest]).to(torch_device_str_nn2)
    y_test_subset_t = torch.from_numpy(y_train_test_subset[ntrain:ntrain+ntest]).long().to(torch_device_str_nn2)

    logging.info("X_test_subset_t and y_test_subset_t created")

    dataset_X_y_test = TensorDataset(X_test_subset_t, y_test_subset_t)
    nn2_test_loader = DataLoader(dataset_X_y_test, batch_size=nn2_batch_size, shuffle=True)

    logging.info("dataset_X_y_test created")

    _train_nn2_with_DLs(nn2_train_loader, nn2_test_loader)



def predict_nn2_from_pd(all_pred_pd):
    """
    Runs NN2 fusion predictions(inference) from several probabiliy data volumes
    Data volumes are provided as h5 files format with information in a pandas dataframe
    from predict_NN1 outupt

    By default, it uses CPU to do the inference to not interfeer with NN1 models and data

    # Returns a list of predictions labels in uint8 format

    """
    logging.info("NN2_predict_from_pd()")

    nsets = all_pred_pd["pred_sets"].max()+1
    logging.info(f"nsets: {nsets}")

    nn2_preds = []
    for iset in range(nsets):
        gc.collect()
        logging.info(f"iset:{iset}")

        #data_5d = data_all_np6d[iset]
        data_5d = aggregate_data_from_pd_iset(all_pred_pd,iset)
        logging.info(f"data_all_np5d.shape: {data_5d.shape}")
        
        r2 = nn2_predict_single_vol(data_5d)
        logging.info(f"iset:{iset}, nn2 prediction shape:{r2.shape}")

        nn2_preds.append(r2)
        
        logging.info("NN2 predictions complete.")

        gc.collect()

    return nn2_preds

def nn2_predict_single_vol(data_5d):
    """
    Runs nn2 predictions (MLP fusion) from a datavolume,
    with shape (pred, class, Z,Y,X)

    returns: prediction as a volume (Z,Y,X), and with np.uint8 data type
    """
    logging.info("nn2_predict_single_vol()")

    global torch_device_str_nn2
    assert data_5d.ndim == 5

    s = data_5d.shape
    nelements = int(np.prod(s[2:]))
    p0= data_5d.reshape( (s[0]*s[1], nelements) )
    data_flat_for_mlp= p0.transpose((1,0))

    nelements_256 = int(nelements//256)
    # device and batch sizes, cpu and batch size fallback 
    batchsize_attempts = [ #(torch_device_str_nn2, nelements), # Using gpu is slow
                ("cpu", nelements_256),
                ("cpu", 65535), #Fallback(s)
                ("cpu", 4096)
                ]
    
    res_s=[]
    b_succeed=False
    for at0 in batchsize_attempts:
        torchdev, batchsize = at0
        try:
            logging.info(f"Attempting to run NN2 inference on nelements:{nelements} with torchdev:{torchdev} and batchsize:{batchsize}")
            nn2_model_fusion.to(torchdev)
            nn2_model_fusion.eval()
            res_s=[]

            # topred_tc= torch.from_numpy(data_flat_for_mlp).float().to(torchdev)
            # data_tc_ds = TensorDataset(topred_tc)
            # data_tc_batcher = DataLoader(data_tc_ds, batch_size=batchsize, shuffle=False)
            # with torch.no_grad():
            #     logging.info("Beggining NN2 inference of whole volume")
            #     for data_batch in tqdm(data_tc_batcher):
            #         #res= torch.squeeze(mlp_model(data_multi_preds_probs_np))
            #         pred = nn2_model_fusion(data_batch[0]) #Can't remember why index 0
            #         pred_argmax = torch.argmax(pred,dim=1)
            #         res_s.append(pred_argmax)

            # Not datasets or dataloaders, just plain slicing from numpy to create batches
            with torch.no_grad():
                logging.info("Beggining NN2 inference of whole volume")

                totalsize = data_flat_for_mlp.shape[0]
                for idx in tqdm(range(0, totalsize, batchsize)):
                    id_end =  idx+batchsize
                    if id_end>totalsize:
                        id_end=totalsize

                    data_batch_np = data_flat_for_mlp[idx:id_end,...]

                    data_batch = torch.from_numpy(data_batch_np).float().to(torchdev)

                    pred = nn2_model_fusion(data_batch)

                    pred_argmax = torch.argmax(pred,dim=1)
                    res_s.append(pred_argmax)
            
            b_succeed=True
        except Exception as e:
            logging.error(f"Error occured, the following exception was throuwn. {str(e)}")
        
        if b_succeed:
            break
    else:
        # This will run if the for loop did not encounter a break
        # which will happend in case of no success in running the inference
        raise RuntimeError(f"Could not run inference with NN2.")
    
    r0 = torch.concatenate(res_s) #Collect all inference batches to a singly flattened array
    r2 = r0.detach().cpu().numpy().astype(np.uint8).reshape(*s[2:])

    return r2


def get_func_from_data_vol_norm_process_str():
    logging.info("get_func_from_data_vol_norm_process_str()")
    logging.info(f"data_vol_norm_process_str:{data_vol_norm_process_str}")
    func0=None
    if data_vol_norm_process_str is not None:
        if data_vol_norm_process_str == "mean_stdev_3":
            func0 = normalise_voldata_to_stdev_3

    return func0

def normalise_volumes(data_vols_list):
    """
    Normalise volumes if process established
    """
    logging.info("normalise_volumes()")
    norm_func = get_func_from_data_vol_norm_process_str()
    
    datavols_list0=None
    if norm_func is None:
        datavols_list0 = data_vols_list
    else:
        logging.info("Normalising data.")
        datavols_list0 = [ norm_func(v0) for v0 in data_vols_list]

    return datavols_list0

def train(datavols_list, labels_list):
    """
    Train NN1 models and NN2 fusion

    Assumes that NN1 models and MLP have been pre-setup

    """
    logging.info("train()")

    if len(NN1_models)==0:
        raise ValueError("No NN1 models. Make sure you set this up first.")
    if nn2_model_fusion is None:
        raise ValueError("No NN2_model_fusion. Make sure you set this up first.")

    datavols_list0 = normalise_volumes(datavols_list)

    assert datavols_list is not None

    train_nn1(datavols_list0, labels_list)

    logging.info("Next train stage is to run 12-way predictions, and use prediction data to train NN2")
    tempdir_pred= tempfile.TemporaryDirectory()
    path_out_results = Path(tempdir_pred.name)
    logging.info(f"tempdir_pred_path:{path_out_results}")

    res_pds = predict_nn1(datavols_list0, path_out_results)
    logging.info("Predict_nn1 complete. This is the resulting dataframe")
    logging.info(str(res_pds))

    logging.info("Passing the predicitons to NN2 for training")

    data_all_np6d = aggregate_data_from_pd(res_pds)

    train_nn2(data_all_np6d, labels_list)

    logging.info("NN1 and NN2 training complete. Don't forget to save model.")

def predict(datavols_list):
    """
    Alias of of predict_from_data_list()
    """
    return predict_from_data_list(datavols_list)

def predict_from_data_list(datavols_list):
    """
    predict, NN1 followed by NN2

    Assumes datavols have not been normalised so normalise them here before running the DL

    NN1 predictions dataframe is stored in global variable last_nn1_prediction_df

    Returns: prediction volume with classes as int values
    """
    
    logging.info("predict_from_data_list()")
    
    # if input is not a list of volumes, turn to a list with one element
    data_in =datavols_list
    if not isinstance(datavols_list,list):
        raise ValueError("datavols_list is not a list of data")

    #Normalise volumes
    datavols_list0 = normalise_volumes(data_in)

    #Creates a temporary folder (will delete after leaving this function!)
    tempdir_pred= tempfile.TemporaryDirectory()
    path_out_results = Path(tempdir_pred.name)
    logging.info(f"tempdir_pred_path:{path_out_results}")

    nn1_prediction_df = predict_nn1(datavols_list0, path_out_results)

    #Clear RAM before next stage
    gc.collect()

    nn2_preds = predict_nn2_from_pd(nn1_prediction_df)

    return nn2_preds

def predict_single(datavol):
    """
    predict, NN1 followed by NN2

    Works with a single 3D data volume

    """
    logging.info("predict_single()")

    if isinstance(datavol, list):
        raise ValueError("datavol is a list. You should use predict_from_data_list() instead")
    
    logging.info(f"datavol.shape:{datavol.shape}")

    # if input is not a list of volumes, turn to a list with one element
    data_in = [datavol]

    #Normalise volumes
    datavols_list0 = normalise_volumes(data_in)

    #Creates a temporary folder (will delete after leaving this function!)
    tempdir_pred= tempfile.TemporaryDirectory()
    path_out_results = Path(tempdir_pred.name)
    logging.info(f"tempdir_pred_path:{path_out_results}")


    nn1_prediction_df = predict_nn1(datavols_list0, path_out_results)

    nn2_preds = predict_nn2_from_pd(nn1_prediction_df)

    if len(nn2_preds)!=1:
        raise ValueError("Result nn2_preds does not have a single element")

    return nn2_preds[0]

def save_lgsegm2_model(fn_out):
    """
    Save model
    fn_out: completed filename with path for saving multi-model segm2


    """

    nn1_models_state_dict = [ m.state_dict() for m in NN1_models]

    
    train_info = f"""
nn1_loss_criterion: {nn1_loss_criterion}
nn1_eval_metric: {nn1_eval_metric}
nn1_lr: {nn1_lr}
nn1_max_lr: {nn1_max_lr}
nn1_train_epochs: {nn1_train_epochs}
nn1_train_CEloss_weights: {nn1_train_CEloss_weights}

nn1_batch_size = {nn1_batch_size}
nn1_num_workers = {nn1_num_workers}

nn2_ntrain: {nn2_ntrain}
nn2_train_epochs: {nn2_train_epochs}
nn2_batch_size: {nn2_batch_size}
nn2_lr: {nn2_lr}
nn2_max_lr: {nn2_max_lr}
nn2_train_do_class_balance: {nn2_train_do_class_balance}
nn2_ntrain_in_class_balance: {nn2_ntrain_in_class_balance}
nn2_train_CEloss_weights: {nn2_train_CEloss_weights}

"""
    
    dict_to_save={
    "nn1_models_class_generator": nn1_models_class_generator,
    "nn1_axes_to_models_indices": nn1_axes_to_models_indices,
    "data_vol_norm_process_str": data_vol_norm_process_str,
    "NN1_models_state_dict": nn1_models_state_dict,

    "nn2_MLP_model_class_generator": nn2_MLP_model_class_generator,
    "NN2_model_dict":nn2_model_fusion.state_dict(),

    "train_info": train_info
    }

    torch.save(dict_to_save, fn_out)


def load_lgsegm2_model(fn):
    """
    Loads lgsegm2 model
    """
    global nn1_models_class_generator
    global nn1_axes_to_models_indices
    global data_vol_norm_process_str
    global nn2_MLP_model_class_generator

    global NN1_models
    global nn2_model_fusion

    logging.info(f"load_lgsegm2_model(), with file {fn}")
    load_model = torch.load(fn)

    nn1_models_class_generator =        load_model["nn1_models_class_generator"]
    nn1_axes_to_models_indices =        load_model["nn1_axes_to_models_indices"]
    data_vol_norm_process_str =         load_model["data_vol_norm_process_str"]
    nn2_MLP_model_class_generator =     load_model["nn2_MLP_model_class_generator"]
    NN1_models_state_dict =             load_model["NN1_models_state_dict"]
    NN2_model_dict =                    load_model["NN2_model_dict"]

    logging.info("Loading NN1 models")
    #update_NN1_models_from_generators()

    for cg0 in nn1_models_class_generator:
        cg0['encoder_weights'] = None # Ensure no weights are preloaded
    
    update_nn1_models_from_generators()

    assert len(NN1_models) == len(NN1_models_state_dict)

    for i, nn1_w in enumerate(NN1_models_state_dict):
        m=NN1_models[i]
        m.load_state_dict(nn1_w)
        m.to(torch_device_str)


    logging.info("Loading NN2 model")

    update_nn2_model_from_generator()
    
    nn2_model_fusion.load_state_dict(NN2_model_dict)
    nn2_model_fusion.to(torch_device_str_nn2)

    logging.info("Loading model complete.")





# Shortcut functions to create train and create models
def quick_new_and_train_one_unet_model_per_axis(datavols_list, labels_list):
    global nn1_models_class_generator
    global nn2_MLP_model_class_generator
    global nn1_train_epochs
    logging.info("quick_new_and_train_one_unet_model_per_axis")

    nn1_models_class_generator= nn1_models_class_generator_default
    nn2_MLP_model_class_generator= nn2_MLP_model_class_generator_default
    # Default 3 unet models, one per axis. 3 classes
    # NN2, MLP 10,10

    #nn1_train_epochs= 10

    update_nn1_models_from_generators()
    update_nn2_model_from_generator()

    train(datavols_list, labels_list)

    logging.info("Training complete")


def quick_new_and_train_2unets_z_xy_models(datavols_list, labels_list):
    global nn1_models_class_generator
    global nn1_axes_to_models_indices
    global nn2_MLP_model_class_generator
    global nn1_train_epochs

    logging.info("quick_new_and_train_one_unet_model_per_axis")

    nn1_models_class_generator= [nn1_dict_gen_default,
    nn1_dict_gen_default.copy()]
    
    nn1_axes_to_models_indices = [0,1,1]

    nn2_MLP_model_class_generator= nn2_MLP_model_class_generator_default
    # Default 3 unet models, one per axis. 3 classes
    # NN2, MLP 10,10

    #nn1_train_epochs= 10

    update_nn1_models_from_generators()
    update_nn2_model_from_generator()

    train(datavols_list, labels_list)

    logging.info("Training complete")


def quick_new_and_train_single_unet_for_all_axis(datavols_list, labels_list):
    global nn1_models_class_generator
    global nn1_axes_to_models_indices
    global nn2_MLP_model_class_generator
    global nn1_train_epochs

    logging.info("quick_new_and_train_single_unet_for_all_axis")

    nn1_models_class_generator= [nn1_dict_gen_default]
    
    nn1_axes_to_models_indices = [0,0,0]

    nn2_MLP_model_class_generator= nn2_MLP_model_class_generator_default
    # Default 3 unet models, one per axis. 3 classes
    # NN2, MLP 10,10

    #nn1_train_epochs= 10

    update_nn1_models_from_generators()
    update_nn2_model_from_generator()

    train(datavols_list, labels_list)

    logging.info("Training complete")

def fuse_max_prob(data_5d ):
    """
    Fuse several prediction probability volumes to labels
    using maximum probability

    data_5d must have typical shape format of (pred, class, Z,Y,X)

    Returns: a volume with uint8 values corresponding to the
    prediction class number for each voxel
    
    """
    data_reduced_along_preds_axis = np.max(data_5d, axis=0)
    # Result will have one axis that disappears, the pred axis
    # Result shape will be (class, Z,Y,X)

    max_label_vol = np.argmax(data_reduced_along_preds_axis, axis=0)
    # result shape will be (Z,Y,X)

    return max_label_vol.astype(np.uint8)

def fuse_max_prob_from_pd(all_pred_pd):
    logging.info("fuse_max_prob_from_pd()")

    nsets = all_pred_pd["pred_sets"].max()+1
    logging.info(f"nsets: {nsets}")

    nn2_preds = []
    for iset in range(nsets):
        gc.collect()
        logging.info(f"iset:{iset}")

        #data_5d = data_all_np6d[iset]
        data_5d = aggregate_data_from_pd_iset(all_pred_pd,iset)
        logging.info(f"data_all_np5d.shape: {data_5d.shape}")
        
        r2 = fuse_max_prob(data_5d)
        logging.info(f"iset:{iset}, max prob fusion result shape:{r2.shape}")

        nn2_preds.append(r2)
        
        logging.info("Max probability fusion complete.")

        gc.collect()

    return nn2_preds


def predict_from_data_list_using_max_prob_fusion(datavols_list):
    """
    Does NN1 predictions followed by fusion
    but instead of fusing volumes using MLP it simply uses maximum probability

    This emulates the volume-segmantics behaviour

    NN1 followed by max prob fusion to collect label

    Returns: prediction volume with classes as uint8 values

    """

    # if input is not a list of volumes, turn to a list with one element
    data_in = datavols_list
    if not isinstance(datavols_list,list):
        raise ValueError("datavols_list is not a list of data")

    #Normalise volumes
    datavols_list0 = normalise_volumes(data_in)

    #Creates a temporary folder (will delete after leaving this function!)
    tempdir_pred= tempfile.TemporaryDirectory()
    path_out_results = Path(tempdir_pred.name)
    logging.info(f"tempdir_pred_path:{path_out_results}")

    nn1_prediction_df = predict_nn1(datavols_list0, path_out_results)

    #Clear RAM before next stage
    gc.collect()

    fusion_preds = fuse_max_prob_from_pd(nn1_prediction_df)

    return fusion_preds

