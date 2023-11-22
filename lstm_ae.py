
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

def_lstm_kwargs = {
            ## output activation
            "activation":"tanh",
            ## cell state activation
            "recurrent_activation":"sigmoid",
            ## initial activation for 'previous output'
            "kernel_initializer":"glorot_uniform",
            ## initial activation for  'previous cell state'
            "recurrent_initializer":"orthogonal",
            "kernel_regularizer":None,
            ## Between-cell cell state dropout rate
            "dropout":0.0,
            ## Between-cell previous output dropout rate
            "recurrent_dropout":0.0,
            }

def get_lstm_stack(name:str, layer_input:Layer, node_list:list, return_seq,
                   bidirectional:False, batchnorm=True, dropout_rate=0.0,
                   lstm_kwargs={}):
    """
    Returns a Layer object after adding a LSTM sequence stack

    :@param name: Unique string name identifying this entire LSTM stack.
    :@param layer_input: layer recieved by this LSTM stack. Typically
        expected to have axes like (batch, sequence, features).
    :@param node_list: List of integers describing the number of nodes in
        subsequent layers starting from the stack input, for example
        [32,64,64,128] would map inputsequences with 32 elements
    """
    lstm_kwargs = {**def_lstm_kwargs.copy(), **lstm_kwargs}
    l_prev = layer_input
    for i in range(len(node_list)):
        ## Intermediate LSTM layers must always return sequences in order
        ## to stack; this is only False if return_seq is False and the
        ## current LSTM layer is the LAST one.
        rseq = (not (i==len(node_list)-1), True)[return_seq]
        tmp_lstm = LSTM(units=node_list[i], return_sequences=rseq,
                        **lstm_kwargs, name=f"{name}_lstm_{i}")
        ## Add a bidirectional wrapper if requested
        if bidirectional:
            tmp_lstm = Bidirectional(
                    tmp_lstm, name=f"{name}_bd_{i}")
        l_new = tmp_lstm(l_prev)
        if batchnorm:
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_new)
        ## Typically dropout is best after batch norm
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new
    return l_prev

def basic_lstmae(
        seq_len:int, feat_len:int, enc_nodes:list, dec_nodes:list, latent:int,
        latent_activation="sigmoid", dropout_rate=0.0, batchnorm=True,
        mask_val=None, bidirectional=True, enc_lstm_kwargs={},
        dec_lstm_kwargs={}):
    """
    Basic LSTM sequence encoder/decoder with optional masking

    Inputs to a seq->seq model like this one are generally assumed to
    be shaped like (N, P, F) for N sequence samples, P points in each sequence,
    and F features per point.

    :@param seq_len: Size of sequence element dimension of input (2nd dim)
    :@param feat_len: Size of feature dimension of input tensor (3rd dim)
    :@param enc_nodes: List of integers corresponding to the cell state and
        hidden state size of the corresponding LSTM layers.
    :@param dec_nodes: Same as above, but for the decoder.
    :@param enc_lstm_kwargs: arguments passed directly to the LSTM layer on
        initialization; use this to change activation, regularization, etc.
    """
    ## Fill any default arguments with the user-provided ones
    enc_lstm_kwargs = {**def_lstm_kwargs.copy(), **enc_lstm_kwargs}
    dec_lstm_kwargs = {**def_lstm_kwargs.copy(), **dec_lstm_kwargs}

    ## Input is like (None, sequence size, feature count)
    l_seq_in = Input(shape=(seq_len, feat_len))

    ## Add a masking layer if a masking value is set
    l_prev = l_seq_in if mask_val is None \
            else Masking(mask_value=mask_val)(l_seq_in)

    ## Do a pixel-wise projection up to the LSTM input dimension.
    ## This seems like a semi-common practice before sequence input,
    ## especially for word embeddings.
    l_prev = TimeDistributed(
            Dense(enc_nodes[0], name="in_projection"),
            name="in_dist"
            )(l_prev)

    ## Add the encoder's LSTM layers
    l_enc_stack = get_lstm_stack(
            name="enc",
            layer_input=l_prev,
            node_list=enc_nodes,
            return_seq=False,
            bidirectional=bidirectional,
            batchnorm=batchnorm,
            lstm_kwargs=dec_lstm_kwargs,
            dropout_rate=dropout_rate
            )

    ## Encode to the latent vector
    l_enc_out = Dense(latent, activation=latent_activation,
                      name="latent_projection")(l_enc_stack)

    ## Copy the latent vector along the output sequence
    l_dec_in = RepeatVector(seq_len)(l_enc_out)

    ## Add decoder's LSTM layers
    l_dec_stack = get_lstm_stack(
            name="dec",
            layer_input=l_dec_in,
            node_list=dec_nodes,
            return_seq=True,
            bidirectional=bidirectional,
            batchnorm=batchnorm,
            lstm_kwargs=dec_lstm_kwargs,
            )

    ## Uniform transform from LSTM output to pixel distribution
    l_dist = Dense(feat_len, activation="linear", name="out_projection")
    l_dec_out = TimeDistributed(l_dist, name="out_dist")(l_dec_stack)

    ## Get instances of Model objects for each autoencoder component.
    ## Each instance correspond to the same weights per:
    ## https://keras.io/api/models/model/
    full = Model(l_seq_in, l_dec_out)
    #encoder = Model(l_seq_in, l_enc_out)
    #decoder = Model(l_enc_out, l_dec_out)
    return full#, encoder, decoder

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

def simple_tv_split(zarr_array, feature_idxs, total_size,
        training_ratio=.8, seed=None):
    """
    Shuffles the provided zarr_array along the first axis, and returns training
    and validation numpy arrays with first-axis sizes summing to total_size.
    Only features (last axis) with the provided indeces will be included.
    """
    idxs = np.arange(zarr_array.shape[0])
    if not seed is None:
        np.random.default_rng(seed).shuffle(idxs)
    else:
        np.random.shuffle(idxs)
    cutoff_idx = int(total_size*training_ratio)
    t_idx = idxs[:cutoff_idx]
    v_idx = idxs[cutoff_idx:total_size]
    T = zarr_array.oindex[t_idx,:,feature_idxs]
    V = zarr_array.oindex[v_idx,:,feature_idxs]
    return T, V

if __name__=="__main__":
    ceres_zarr_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    modis_val_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")
    modis_train_path = Path("/rstor/mdodson/aes770hw4/modis_training.zip")

    ## Directory with sub-directories for each model.
    model_parent_dir = Path("data/models/")

    modis_train = zarr.Array(
            store=zarr.storage.ZipStore(modis_train_path.as_posix(),mode="r"),
            read_only=True,
            )
    modis_val = zarr.Array(
            store=zarr.storage.ZipStore(modis_val_path.as_posix(),mode="r"),
            read_only=True,
            )
    print(modis_train.shape, modis_val.shape)

    ## Identifying label for this model
    model_name= "lstmae_1"
    ## Size of batches in samples
    batch_size = 128
    ## Batches to draw asynchronously from the generator
    batch_buffer = 2
    ## Seed for subsampling training and validation data
    rand_seed = 20231121
    ## Indeces of features to train on
    modis_feat_idxs = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,24)

    ## Make the directory for this model run, ensuring no name collision.
    model_dir = model_parent_dir.joinpath(model_name)
    assert not model_dir.exists()
    model_dir.mkdir()

    ## Define callbacks for model progress tracking
    c_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8)
    c_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss", save_best_only=True,
            filepath=model_dir.joinpath(
                model_name+"_{epoch}_{mse:.2f}.hdf5"))
    c_csvlog = tf.keras.callbacks.CSVLogger(
            model_dir.joinpath("prog.csv"))

    ## Build a lstm autoencoder
    model = basic_lstmae(
            seq_len=400,
            feat_len=len(modis_feat_idxs),
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
            )#[0]

    ## Write a model summary to stdout and to a file
    model.summary()
    with model_dir.joinpath(model_name+"_summary.txt").open("w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print(f"Compiling model")
    model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="mse",
            metrics=["mse"],
            )

    print(f"Making generators")
    ## Construct generators for the training and validation data
    train_gen = tf.data.Dataset.from_generator(
            shuffle_generator,
            args=(modis_train_path.as_posix(), rand_seed, modis_feat_idxs),
            output_signature=(
                tf.TensorSpec(shape=(400,22), dtype=tf.float16),
                tf.TensorSpec(shape=(400,22), dtype=tf.float16),
                ))

    val_gen = tf.data.Dataset.from_generator(
            shuffle_generator,
            args=(modis_val_path.as_posix(), rand_seed, modis_feat_idxs),
            output_signature=(
                tf.TensorSpec(shape=(400,22), dtype=tf.float16),
                tf.TensorSpec(shape=(400,22), dtype=tf.float16),
                ))

    print(f"Fitting model")
    ## Train the model on the generated tensors
    hist = model.fit(
            train_gen.batch(batch_size).prefetch(batch_buffer),
            epochs=400,
            ## Number of batches to draw per epoch. Use full dataset by default
            #steps_per_epoch=modis_train.shape[0]//batch_size,
            steps_per_epoch=100, ## 12,800 training samples per epoch
            validation_data=val_gen.batch(batch_size).prefetch(batch_buffer),
            ## batches of validation data to draw per epoch
            validation_steps=10, ## 1,280 validation samples per epoch
            validation_freq=1, ## Report validation loss each epoch
            callbacks=[
                c_early_stop,
                c_checkpoint,
                c_csvlog,
                ],
            verbose=2,
            )
    model.save(model_dir.joinpath(model_name+".keras"))
