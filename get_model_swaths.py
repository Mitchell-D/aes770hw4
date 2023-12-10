
from pathlib import Path
#from swath_shapes import swath_validation, swath_training
from random import random
import pickle as pkl
from multiprocessing import Pool

#from FG1D import FG1D
import zarr
import numpy as np
import os

#import keras_tuner
import tensorflow as tf

from norm_vals import ceres_means, ceres_stdevs, modis_means, modis_stdevs
from lstm_ae import get_agg_loss

ceres_means = dict(ceres_means)
ceres_stdevs = dict(ceres_stdevs)
modis_means = dict(modis_means)
modis_stdevs = dict(modis_stdevs)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus):
    ##tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

def single_swath_mask(z_ceres, epoch_feat_idx):
    epochs = z_ceres[...,epoch_feat_idx]
    epochs = (epochs*ceres_stdevs["epoch"]+ceres_means["epoch"]).astype(int)
    cur_time = 0
    break_points = []
    for i in range(epochs.shape[0]):
        if abs(epochs[i]-cur_time) > 300:
            if cur_time != 0:
                break_points.append(i)
            cur_time = epochs[i]
    break_points.append(epochs.shape[0])
    return break_points

def swath_to_pkl(model_path, ceres_zarr_path, modis_zarr_path,
        idx0, idxf, out_dir):
    ## Indeces of features to train on
    modis_band_idxs = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
    modis_space_idxs = (19,20,21,22,23,24)

    ## Retrieve the encoder from the model
    agg_loss = get_agg_loss(band_ratio=.7, ceres_band_cutoff_idx=2)
    model = tf.keras.models.load_model(
            model_path,custom_objects={"agg_loss":agg_loss})

    ceres_zarr = zarr.open(ceres_zarr_path, mode="r")
    modis_zarr = zarr.open(modis_zarr_path, mode="r")

    ## first swath only
    cswath = ceres_zarr[idx0:idxf]
    mswath = modis_zarr[idx0:idxf]

    ## run the model on the single swath
    pred = model(mswath[...,list(modis_band_idxs)+list(modis_space_idxs)])
    ## all the sequence values are reversed per my LSTM tradition, I guess.
    pred = pred[:,::-1]

    ## splice true modis lat/lon values with model output
    ## Original CERES fluxes like (N,1)
    clat = cswath[...,0]*ceres_stdevs["lat"]+ceres_means["lat"]
    clon = cswath[...,1]*ceres_stdevs["lon"]+ceres_means["lon"]
    csw = cswath[...,5]*ceres_stdevs["swflux"]+ceres_means["swflux"]
    clw = cswath[...,6]*ceres_stdevs["lwflux"]+ceres_means["lwflux"]
    epoch = np.round(np.average(cswath[...,-1]) \
            * ceres_stdevs["epoch"] + ceres_means["epoch"]).astype(int)
    ## MODIS lat/lons with shape (N, 400)
    mlat = np.ravel(mswath[...,16]) * modis_stdevs["lat"] + modis_means["lat"]
    mlon = np.ravel(mswath[...,17]) * modis_stdevs["lon"] + modis_means["lon"]
    ## Model outputs w/ shape (N, 400)
    ## Fine model fluxes (not averaged per sample)
    pfsw = np.ravel(pred[...,0]) \
            * ceres_stdevs["swflux"] + ceres_means["swflux"]
    pflw = np.ravel(pred[...,1]) \
            * ceres_stdevs["lwflux"] + ceres_means["lwflux"]
    ## Coarse model fluxes (averaged per sample)
    pcsw = np.average(pred[...,0],axis=1) \
            * ceres_stdevs["swflux"] + ceres_means["swflux"]
    pclw = np.average(pred[...,1],axis=1) \
            * ceres_stdevs["lwflux"] + ceres_means["lwflux"]

    ## Make 1d feature grids for CERES fluxes and the model coarse/fine flux
    pfine = (["lat", "lon", "sw", "lw"],
        list(map(np.squeeze, [mlat, mlon, pfsw, pflw])))
    pcoarse = (["lat", "lon", "sw", "lw"],
        list(map(np.squeeze, [clat, clon, pcsw, pclw])))
    ceres = (["lat", "lon", "sw", "lw"],
        list(map(np.squeeze, [clat, clon, csw, clw])))
    pkl_path = out_dir.joinpath(f"pred_{epoch}.pkl")
    pkl.dump((ceres, pcoarse, pfine), pkl_path.open("wb"))
    return pkl_path

def _swath_to_pkl(a):
    return swath_to_pkl(*a)

def mp_swath_to_pkl(model_path, ceres_zarr_path, modis_zarr_path,
        out_dir, workers):
    z_ceres = zarr.open(ceres_zarr_path, mode="r")
    bps = single_swath_mask(z_ceres, epoch_feat_idx=-1)
    del z_ceres
    idx_combos = [(bps[i],bps[i+1]) for i in range(len(bps)-1)]
    args = [(model_path, ceres_zarr_path, modis_zarr_path, *idx_pair, out_dir)
            for idx_pair in idx_combos]
    with Pool(workers) as pool:
        for out_path in pool.imap(_swath_to_pkl, args):
            print(f"Loaded swath output to {out_path.as_posix()}")

if __name__=="__main__":
    ceres_val_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    ceres_train_path = Path("/rstor/mdodson/aes770hw4/ceres_training.zip")
    ceres_test_path = Path("/rstor/mdodson/aes770hw4/ceres_testing.zip")
    modis_val_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")
    modis_train_path = Path("/rstor/mdodson/aes770hw4/modis_training.zip")
    modis_test_path = Path("/rstor/mdodson/aes770hw4/modis_testing.zip")

    ## Directory with sub-directories for each model.
    model_parent_dir = Path("data/models/")
    #ed_path = model_parent_dir.joinpath("lstmed_4/lstmed_4_93.hdf5")
    ed_path = model_parent_dir.joinpath("lstmed_1/lstmed_1.keras")
    swath_out_dir = Path("/rstor/mdodson/aes770hw4/output_1")

    workers = 8
    ## Seed for subsampling training and validation data
    rand_seed = 20231121
    ceres_feat_idxs = (5,6) ## (swflux, lwflux)

    mp_swath_to_pkl(ed_path, ceres_test_path, modis_test_path,
            swath_out_dir, workers)
