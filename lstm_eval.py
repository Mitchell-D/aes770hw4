
from pathlib import Path
#from swath_shapes import swath_validation, swath_training
from random import random
import pickle as pkl

#from FG1D import FG1D
import zarr
import numpy as np
import os

#import keras_tuner
import tensorflow as tf

from norm_vals import ceres_means, ceres_stdevs, modis_means, modis_stdevs

def shuffle_generator(zarr_path, seed, feature_idxs):
    """
    Generator yielding random first-dimension samples from a large zarr array.
    """
    zarr_memmap = zarr.open(zarr_path.decode('ASCII'), mode="r")
    idxs = np.arange(zarr_memmap.shape[0])
    np.random.default_rng(seed).shuffle(idxs)
    for i in idxs:
        tmpx = zarr_memmap[i]
        tmpx = tmpx[...,feature_idxs]
        tmpy = np.copy(tmpx)[:,::-1]
        yield tuple(map(tf.convert_to_tensor, (tmpx, tmpy)))

def chunk_evaluate(model, in_zarr_path:Path, out_zarr_path:Path,
        replace=False, chunk_size=800, feature_idxs=None):
    """ """
    if not replace:
        assert not out_zarr_path.exists()
    z_in = zarr.open(in_zarr_path, mode="r")
    #z_out = zarr.creation.open_like(z_in, out_zarr_path, mode="w")
    out_shape = z_in.shape if feature_idxs is None \
            else (*z_in.shape[:2], len(feature_idxs))
    z_out = zarr.creation.create(
            shape=out_shape,
            chunks=(1,*out_shape[1:]),
            store=out_zarr_path.as_posix(),
            )
    full_chunks = z_in.shape[0]//chunk_size
    for i in range(full_chunks):
        X = z_in.oindex[i*chunk_size:(i+1)*chunk_size]
        if not feature_idxs is None:
            X = X[...,feature_idxs]
        Z = model(X, training=False).numpy()
        z_out.oindex[i*chunk_size:(i+1)*chunk_size] = Z
        print(f"Finished {i*chunk_size}-{(i+1)*chunk_size}")
    #Z = model(z_in.oindex[full_chunks*chunk_size:], training=False)
    if full_chunks*chunk_size > z_in.shape[0]:
        X = z_in.oindex[full_chunks*chunk_size:]
        if not feature_idxs is None:
            X = X[...,feature_idxs]
        Z = model(X, training=False).numpy()
        z_out.oindex[full_chunks*chunk_size:] = Z
    z_out.flush()

if __name__=="__main__":
    ceres_path = Path("/rstor/mdodson/aes770hw4/ceres_testing.zip")
    modis_path = Path("/rstor/mdodson/aes770hw4/modis_testing.zip")
    eval_path = Path("/rstor/mdodson/aes770hw4/eval_testing.zip")

    ## Directory with sub-directories for each model.
    model_parent_dir = Path("data/models/")
    model_path = model_parent_dir.joinpath("lstmae_1/lstmae_1_9_0.19.hdf5")

    ## Identifying label for this model
    model_name= "lstmae_14"
    ## Size of batches in samples
    batch_size = 32
    ## Batches to draw asynchronously from the generator
    batch_buffer = 4
    ## Seed for subsampling training and validation data
    rand_seed = 20231121
    ## Indeces of features to train on
    modis_feat_idxs = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,24)
    #modis_feat_idxs = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

    model = tf.keras.models.load_model(model_path)
    chunk_evaluate(model, modis_path, eval_path, feature_idxs=modis_feat_idxs)
