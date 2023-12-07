
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
from tensorflow.keras.layers import Layer,Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus):
    ##tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
## Moved to lstm_ae.py
@tf.autograph.experimental.do_not_convert
def agg_loss(y_true, y_pred):
    """ """
    ## Average all of the sequence outputs to get the prediction
    t_bands = y_true[:,0,:2] ## CERES bands are just copied along ax0
    p_bands = y_pred[:,:,:2] ## Predicted bands should vary
    t_space = y_true[:,:,2:]
    p_space = y_pred[:,:,2:]
    ## Take the average of all model predictions in the footprint
    p_bands = tf.math.reduce_mean(p_bands, axis=1)
    ## MSE independently for spatial and lw/sw predictions
    L_bands = tf.math.reduce_mean(tf.square(t_bands-p_bands))
    L_space = tf.math.reduce_mean(tf.square(t_bands-p_bands))
    #return .8*L_bands+.2*L_space ## lstmed_0
    #return 1*L_bands+0*L_space ## lstmed_1
    return .5*L_bands+.5*L_space ## lstmed_2

def ceres_generator(modis_zarr_path, ceres_zarr_path, seed,
        modis_band_idxs, modis_space_idxs, ceres_feature_idxs):
    """
    Generator for data to upsample modis to ceres resolution
    """
    modis = zarr.open(modis_zarr_path.decode('ASCII'), mode="r")
    ceres = zarr.open(ceres_zarr_path.decode('ASCII'), mode="r")
    #modis = zarr.open(modis_zarr_path, mode="r")
    #ceres = zarr.open(ceres_zarr_path, mode="r")
    idxs = np.arange(modis.shape[0])
    np.random.default_rng(seed).shuffle(idxs)
    for i in idxs:
        tmpx = modis[i]
        ## note the reliance on the fact that modis bands must uniformly
        ## precede the spatial components. In other words, the feature
        ## order that the encoder was trained on must be equivalent to
        ## modis_band_idxs + modis_space_idxs
        X = tmpx[...,list(modis_band_idxs)+list(modis_space_idxs)]
        #tmpy = np.copy(tmpx)[:,::-1]
        tmpy = ceres[i]
        ## Ceres fluxes only
        tmpy = np.tile(tmpy[...,ceres_feature_idxs], (X.shape[0],1))
        ## Append the spatial information to the ceres fluxes
        ## Still flipping the output for consistency
        Y = np.hstack((tmpy, tmpx[...,modis_space_idxs]))[:,::-1]
        yield tuple(map(tf.convert_to_tensor, (X, Y)))
'''

if __name__=="__main__":
    ceres_val_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    ceres_train_path = Path("/rstor/mdodson/aes770hw4/ceres_training.zip")
    modis_val_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")
    modis_train_path = Path("/rstor/mdodson/aes770hw4/modis_training.zip")

    ## Directory with sub-directories for each model.
    model_parent_dir = Path("data/models/")
    #ae_path = model_parent_dir.joinpath("lstmae_1/lstmae_1_9_0.19.hdf5")
    ae_path = model_parent_dir.joinpath("lstmae_1/lstmae_1.keras")

    ## Identifying label for this model
    ed_name= "lstmed_5"
    ## Size of batches in samples
    batch_size = 32
    ## Batches to draw asynchronously from the generator
    batch_buffer = 4
    ## Seed for subsampling training and validation data
    rand_seed = 20231121
    ## Indeces of features to train on
    modis_band_idxs = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
    modis_space_idxs = (19,20,21,22,23,24)
    ceres_feat_idxs = (5,6) ## (swflux, lwflux)
    #modis_feat_idxs = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

    ## Retrieve the encoder from the model
    model = tf.keras.models.load_model(ae_path)
    encoder = Model(model.input, model.get_layer("latent_projection").output)

    from lstm_ae import lstm_decoder, get_agg_loss, ceres_generator

    ed = lstm_decoder(
            encoder=encoder,
            seq_len=400,
            feat_len=8,
            dec_nodes=[32, 64, 128],
            bidirectional=True,
            batchnorm=True,
            dropout_rate=0.2,
            dec_lstm_kwargs={},
            )

    agg_loss = get_agg_loss(band_ratio=.7, ceres_band_cutoff_idx=2)

    #'''
    ## Make the directory for this model run, ensuring no name collision.
    ed_dir = model_parent_dir.joinpath(ed_name)
    assert not ed_dir.exists()
    ed_dir.mkdir()

    ## Define callbacks for model progress tracking
    c_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50)
    c_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss", save_best_only=True,
            filepath=ed_dir.joinpath(
                ed_name+"_{epoch}.hdf5"))
    c_csvlog = tf.keras.callbacks.CSVLogger(
            ed_dir.joinpath("prog.csv"))

    ## Write a model summary to stdout and to a file
    ed.summary()
    with ed_dir.joinpath(ed_name+"_summary.txt").open("w") as f:
        ed.summary(print_fn=lambda x: f.write(x + '\n'))
    #'''


    '''
    cg = ceres_generator(modis_train_path.as_posix(),ceres_train_path.as_posix(),
                rand_seed, modis_band_idxs, modis_space_idxs, ceres_feat_idxs)
    out = next(cg)
    print(out[0].shape, out[1].shape)
    '''

    print(f"Compiling encoder-decoder")
    ed.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=agg_loss,
            #metrics=["mse"],
            )

    print(f"Making generators")
    ## Construct generators for the training and validation data
    train_gen = tf.data.Dataset.from_generator(
            ceres_generator,
            args=(modis_train_path.as_posix(), ceres_train_path.as_posix(),
                rand_seed, modis_band_idxs, modis_space_idxs, ceres_feat_idxs),
            output_signature=(
                tf.TensorSpec(shape=(400,22), dtype=tf.float16),
                tf.TensorSpec(shape=(400,8), dtype=tf.float16),
                ))

    val_gen = tf.data.Dataset.from_generator(
            ceres_generator,
            args=(modis_val_path.as_posix(), ceres_val_path.as_posix(),
                rand_seed, modis_band_idxs, modis_space_idxs, ceres_feat_idxs),
            output_signature=(
                tf.TensorSpec(shape=(400,22), dtype=tf.float16),
                tf.TensorSpec(shape=(400,8), dtype=tf.float16),
                ))

    print(f"Fitting model")
    ## Train the model on the generated tensors
    hist = ed.fit(
            train_gen.batch(batch_size).prefetch(batch_buffer),
            epochs=1000,
            ## Number of batches to draw per epoch. Use full dataset by default
            steps_per_epoch=100, ## 3,200 samples per epoch
            validation_data=val_gen.batch(batch_size).prefetch(batch_buffer),
            ## batches of validation data to draw per epoch
            validation_steps=25, ## 3,200 samples per validation
            validation_freq=1, ## Report validation loss each epoch
            callbacks=[
                c_early_stop,
                c_checkpoint,
                c_csvlog,
               ],
            verbose=2,
            )
    model.save(ed_dir.joinpath(ed_name+".keras"))
