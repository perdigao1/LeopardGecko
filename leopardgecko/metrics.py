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

import numpy as np
#import matplotlib.pyplot as plt #Needed here?
import dask.array as da
import torch
import logging

def MetricScoreOfVols_Accuracy(vol0, vol1, loginfosum=False):
    '''
    Get the whole-volume Accuracy metric between two volumes that have been segmented
    Uses dask array by default
    '''
    logging.info("MetricScoreOfVols_Accuracy()")
    vol0_da = da.array(vol0)
    vol1_da = da.array(vol1)
    #equalvol = da.equal(vol0_da,vol1_da).astype(np.float32)
    equalvol = da.equal(vol0_da,vol1_da).astype(np.uint8) #reduce memory footprint
    
    if loginfosum:
        sum = da.sum(equalvol, dtype=np.float64).compute() #compute it float64 accumulator
        logging.info(f"For interest sum = {sum}, volsize = {equalvol.size}")

    res = da.mean(equalvol, dtype=np.float64) #Compute mean using float64 accumulator precision

    res_np=res.compute()

    return res_np


def MetricScoreOfVols_Accuracy1(vol0, vol1, showinfosum=False):
    '''
    Alternative
    Get the whole-volume Accuracy metric between two volumes that have been segmented
    '''
    logging.info("MetricScoreOfVols_Accuracy1()")
    vol0_da = da.array(vol0)
    vol1_da = da.array(vol1)
    vol_sub = vol0_da-vol1_da
    equalvol = da.where(vol_sub==0, 1.0, 0.0)

    logging.info("equalvol calculated")

    if showinfosum:
        sum0 = da.sum(equalvol)
        sum = sum0.compute()
        logging.info(f"For interest sum = {sum}, volsize = {equalvol.size}")
    
    res = equalvol.mean().compute()
    return res

def MetricScoreOfVols_get_dicescore(d0,d1, iseg):
    """
    Get dice coefficient between two volumes, for the segmentation given
    
    For a given iseg class, it uses the formula

    d_iclass = 2*(p*t) / (p+t)

    where p is the number of voxels where voxel==iclass in both volumes (intersection)
    v0[z,y,x]==iclass AND v1[z,y,x]==iclass

    and t is the number of voxels where voxels==iclass in any of the volumes (union)
    v0[z,y,x]==iclass OR v1[z,y,x]==iclass


    This function is compatible with numpy or dask.
    If using dask, don't forget to collect value using .compute()

    """

    #Check whether data is numpy or dask
    datamodule=np
    if isinstance(d0,da.Array):
        datamodule=da

    p = datamodule.where(d0 == iseg, 1, 0)
    t = datamodule.where(d1 == iseg, 1, 0)

    # c_inter = (p*t).astype(np.float32).sum()
    # c_union = (p+t).astype(np.float32).sum()
    c_inter = datamodule.sum( (p*t) , dtype=np.float64 )
    c_union = datamodule.sum(p, dtype=np.float64 ) + datamodule.sum(t, dtype=np.float64 )

    dicescore0 = 2.*c_inter/c_union if c_union > 0 else np.nan

    return dicescore0

def MetricScoreOfVols_Dice(vol0, vol1, useBckgnd=False, use_dask=None):
    '''
    Get the whole-volume Multiclass metric between two volumes that have been segmented
    Returns the average dice score and the individual dice coefficients

    '''

    logging.info("MetricScoreOfVols_Dice()")

    #Check arrays have the same shape
    shape1 = vol0.shape
    shape2 = vol1.shape
    if not np.array_equal( np.array(shape1), np.array(shape2)):
        logging.error("Arrays have different shapes. Exiting with None")
        return None

    #Check both volumes dtype is int
    if not ( np.issubdtype(vol0.dtype, np.integer) and np.issubdtype(vol1.dtype, np.integer) ):
        logging.info("Volumes are not integer type. Try to convert them")
        vol0 = vol0.astype(np.int8)
        vol1= vol1.astype(np.int8)

        #equalvol = np.equal(vol0,vol1).astype(np.float32)
        #res = equalvol.mean()

    #Number of segmentations(this is a bad estimate)
    # nseg0 = np.max(vol0)
    # nseg1 = np.max(vol1)
    # logging.info(f"Number of class segmentations in first volume {nseg0+1}")
    # logging.info(f"Number of class segmentations in second volume {nseg1+1}")

    segs0 = np.unique(vol0)
    segs1 = np.unique(vol1)
    nseg0 = len(segs0)
    nseg1 = len(segs1)

    logging.info(f"Number of class segmentations in first volume {nseg0}")
    logging.info(f"Number of class segmentations in second volume {nseg1}")

    if nseg0 != nseg1:
        logging.warning ("Number of segmentations between volumes is different.")
    

    # nseg = max(nseg0, nseg1)

    # isegstart=1
    # if useBckgnd: isegstart=0
    
    #Calculate dice of each metric
    #Similar to code in https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L343

    allsegs = np.unique(np.concatenate((segs0, segs1)))

    #dicescores=np.array([]) #accumulator
    dicescores={} # dictionary rather than list

    #Use dask arrays is arrays are too large >1Gb
    use_dask0=use_dask
    # if vol0.size> 1e9:
    #     logging.info("Arrays too large >1Gb, will use dask.")
    #     usedask=True
    
    if not use_dask0:
        logging.info("Calculating using numpy")
        try:
            for iseg in allsegs:
                if (iseg==0 and useBckgnd) or (iseg!=0):
                    # p = np.where(vol0 == iseg, 1, 0)
                    # t = np.where(vol1 == iseg, 1, 0)
                    # # c_inter = (p*t).astype(np.float32).sum()
                    # # c_union = (p+t).astype(np.float32).sum()
                    # c_inter = np.sum( (p*t) , dtype=np.float64 )
                    # c_union = np.sum( (p+t) , dtype=np.float64 )

                    # #dicescore= 2.*self.inter[c]/self.union[c] if self.union[c] > 0 else np.nan
                    # dicescore= 2.*c_inter/c_union if c_union > 0 else np.nan
                    # dicescores = np.append(dicescores, dicescore)

                    dice0 = MetricScoreOfVols_get_dicescore(vol0, vol1, iseg)
                    #dicescores = np.append(dicescores, dicescore)
                    dicescores[iseg]=dice0
        except:
            logging.info("Error occurred when calculating dicescores. Will try to use dask")
            use_dask0=True

    if use_dask0:
        #Use dask
        #dicescores=np.array([])
        #for iseg in range(isegstart,nseg+1):
        logging.info("Calculating using dask")
        dicescores={} #reset
        vol0_da = da.from_array(vol0)
        vol1_da = da.from_array(vol1)
        for iseg in allsegs:
            if (iseg==0 and useBckgnd) or (iseg!=0):
                
                dice0 = MetricScoreOfVols_get_dicescore(vol0_da, vol1_da, iseg)
                dice0_c = dice0.compute()

                #dicescores = np.append(dicescores, dicescore)
                dicescores[iseg]=dice0_c

    logging.info(f"dicescores = {dicescores}")

    
    #Compute final dicescore
    dicescore_all = np.nanmean(np.array( list(dicescores.values() ) ))

    return dicescore_all, dicescores


def get_metric_stats_per_class(data_lbls,target, nclasses=None):
    """
    Gets true positive, false positive, false negative and true negative sums (tp,fp,fn,tn)
    for each class

    If nclasses is None (default) then it gets for all the classes based in the maximum
    value of both data_lbls and target

    Returns:
    stats_per_class: A list of (tp,fp,fn,tn) values for each class starting at class0
    [ (tp_0,fp_0,fn_0,tn_0) , (tp_1,fp_1,fn_1,tn_1), ...]

    """
    
    if isinstance(data_lbls, np.ndarray):
        data_lbls = torch.from_numpy(data_lbls)

    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    data_flat = torch.ravel(data_lbls)
    target_flat = torch.ravel(target)
    
    #Get max class
    if nclasses is None:
        nclasses = max(int(torch.max(data_flat)), int(torch.max(target_flat)) )+1
        logging.info(f"nclasses:{nclasses}")

    stats_per_class=[]
    for iclass in range(nclasses):
        pred_bin = data_flat==iclass
        gnd_bin =  target_flat==iclass
        tp = int(torch.sum(torch.bitwise_and(pred_bin,gnd_bin)))
        tn = int(torch.sum(torch.bitwise_and(torch.bitwise_not(pred_bin),torch.bitwise_not(gnd_bin))))
        fp = int(torch.sum(torch.bitwise_and(pred_bin,torch.bitwise_not(gnd_bin))))
        fn = int(torch.sum(torch.bitwise_and(torch.bitwise_not(pred_bin),gnd_bin)))
        metr_t= (tp,fp,fn,tn)  # Same order as used in SMP get_stats
        stats_per_class.append(metr_t)

    return stats_per_class

def accuracy_from_stats(tp,fp,fn,tn):
    return (tp+tn)/(tp+tn+fp+fn)

def iou_from_stats(tp,fp,fn,tn):
    return tp/(tp+fp+fn)

def f1dice_from_stats(tp,fp,fn,tn):
    return 2*tp/(2*tp+fp+fn)

def recall_from_stats(tp,fp,fn,tn):
    return tp/(tp+fn)

def precision_from_stats(tp,fp,fn,tn):
    return tp/(tp+fp)