
from pathlib import Path
from random import random
import pickle as pkl

#from FG1D import FG1D
import zarr
import numpy as np
import os

import tensorflow as tf

def shuffle_split(ceres_zarr_path, modis_zarr_path, ceres_dir, modis_dir):
    """
    """
    pass

if __name__=="__main__":
    #ceres_zarr_path = Path("data/buffer/ceres_training.zip")
    #modis_zarr_path = Path("data/buffer/modis_training.zip")
    ceres_zarr_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    modis_zarr_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")

    """ Load swaths from the zarr arrays """
    ceres = zarr.Array(ceres_zarr_path, read_only=True)
    modis = zarr.Array(modis_zarr_path, read_only=True)

    print(ceres.shape, modis.shape)
    print(dict(ceres.attrs), dict(modis.attrs))

    t_ratio = .8
    total_size = 500000
    modis_train_on = np.array((
        0,1,2,3,4,5,6,7,8,9,10,11,12,
        13,14,15,19,20,21,22,23,24
        ))
    shuffle = np.arange(modis.shape[0])
    np.random.shuffle(shuffle)
    cutoff_idx = int(total_size*t_ratio)
    t_idx = shuffle[:cutoff_idx]
    v_idx = shuffle[cutoff_idx:total_size]
    #print(modis.shape, t_idx.shape, modis_train_on.shape)
    T = modis.oindex[t_idx,:,modis_train_on]
    V = modis.oindex[v_idx,:,modis_train_on]
    print(T.shape, V.shape)

    c_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4)
    c_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss", save_best_only=True,
            filepath=buffer_dir.joinpath(
                "lstmae_0_{epoch}_{mse:.2f}.hdf5"))
    c_csvlog = tf.keras.callbacks.CSVLogger(
            buffer_dir.joinpath("lstmae_0_prog.csv"))

    #'''
    model = basic_lstmae(
            seq_len=400,
            feat_len=len(modis_train_on),
            enc_nodes=[64, 64, 64],
            dec_nodes=[64, 64, 64],
            latent=32,
            latent_activation="sigmoid",
            dropout_rate=0.0,
            batchnorm=True,
            mask_val=None,
            bidirectional=True,
            enc_lstm_kwargs={},
            dec_lstm_kwargs={},
            )

    model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="mse",
            metrics=["mse", "val_loss"],
            )

    hist = model.fit(
            x=T,
            y=T[:,::-1],
            epochs=800,
            callbacks=[
                c_early_stop,
                c_checkpoint,
                c_csvlog,
                ],
            validation_data=(V, V[:,::-1]),
            )
    #'''

    '''
    hp = keras_tuner.HyperParameters()
    #model = hp_basic_lstmae(hp)
    #print(model(mdat))
    tuner_dir = Path("data/tuner")
    tuner_dir = Path("data/buffer/train")

    tuner = keras_tuner.Hyperband(
            hp_basic_lstmae,
            objective="val_loss",
            ## Maximum epochs per training run
            max_epochs=40,
            ## reduction factor from https://doi.org/10.48550/arXiv.1603.06560
            factor=3,
            directory=tuner_dir,
            project_name="lstmae_0",
            max_retries_per_trial=1,
            )

    print(tuner.search_space_summary())
    tuner.search(
            T, T[:,::-1],
            validation_data=(V, V[:,::-1]),
            batch_size=32,
            callbacks=[c_early_stop, c_checkpoint],
            )
    print(tuner.get_best_hyperparameters())
    '''

    #print(model.summary())
    #print(encoder.summary())
    #print(decoder.summary())
