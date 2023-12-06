
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
    ceres_val_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    ceres_train_path = Path("/rstor/mdodson/aes770hw4/ceres_training.zip")
    modis_val_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")
    modis_train_path = Path("/rstor/mdodson/aes770hw4/modis_training.zip")

    ## Directory with sub-directories for each model.
    model_parent_dir = Path("data/models/")
    ae_path = model_parent_dir.joinpath("lstmae_1/lstmae_1_9_0.19.hdf5")

    ## Identifying label for this model
    ed_name= "lstmed_2"
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

    from lstm_ae import lstm_decoder

    ed = lstm_decoder(
            encoder=encoder,
            seq_len=400,
            feat_len=2,
            dec_nodes=[64, 64, 64],
            bidirectional=True,
            batchnorm=True,
            dec_lstm_kwargs={},
            )

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

    #cg = ceres_generator(modis_train_path.as_posix(),ceres_train_path.as_posix(),
    #            rand_seed, modis_band_idxs, modis_space_idxs, ceres_feat_idxs)

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
