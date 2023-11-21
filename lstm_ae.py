
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
    encoder = Model(l_seq_in, l_enc_out)
    decoder = Model(l_enc_out, l_dec_out)
    return full, encoder, decoder

'''
def hp_basic_lstmae(hp:keras_tuner.HyperParameters):
    """
    Model constructor function for hyperparamter tuning.

    All search space hyperparameter ranges should be set in this method.

    :@return: 3-tuple of Model object references (full_model,encoder,decoder),
        where the full model has been compiled with hyperparameters.
    """
    hdict = {
            ## Size of input sequence (2nd dimension of batch array)
            "in_seq_len":hp.Fixed("in_seq_len", 400),
            ## Size of feature vector (3nd dimension of batch array)
            "in_feat_len":hp.Fixed("in_feat_len", 25),
            ## List of encoder node counts defining each layer from in to out.
            "enc_lstm_nodes":[
                #hp.Fixed("enc_lstm_0", 64),
                hp.Int("enc_lstm_0", 32, 96, 32),
                hp.Int("enc_lstm_1", 32, 96, 32),
                hp.Int("enc_lstm_2", 32, 96, 32),
                ],
            ## List of decoder node counts defining each layer from in to out
            "dec_lstm_nodes":[
                hp.Fixed("dec_lstm_0", 64),
                hp.Fixed("dec_lstm_0", 64),
                hp.Fixed("dec_lstm_0", 64),
                ],
            ## Size of latent representation vector
            #"latent":hp.Fixed("latent", 32),
            "latent":hp.Int("latent", 32, 32, 96),
            "latent_activation":hp.Fixed("latent_activation", "linear"),
            ## Dropout rate between lstm layers (instead of between seq cells)
            "dropout_rate":hp.Float("dropout_rate", 0.0, 0.3, 3),
            ## Use bidirectional LSTMs for encoder and decoder
            "batchnorm":hp.Fixed("batchnorm",True),
            ## Use bidirectional LSTMs for encoder and decoder
            "bidirectional":hp.Fixed("bidirectional", True),
            ## Define the value that marks sequence vectors as ignorable
            ## when all the vector's features are equal to it.
            "mask_val":hp.Fixed("mask_val", 9999.9999),
            ## Optimizer learning rate
            "learning_rate":hp.Fixed("learning_rate", 0.001),

            ## Dropout rate for encoder cell state
            "enc_lstm-cell_dropout":hp.Fixed(
                "enc_lstm-cell_dropout", 0.0),
            ## Activation function between lstm cell states
            "enc_lstm-cell_activation":hp.Fixed(
                "enc_lstm-cell_activation", "sigmoid"),
            ## Initial weight activation method for encoder cell state
            "enc_lstm-cell_init":hp.Fixed(
                "enc_lstm-cell_init","orthogonal"),

            ## Dropout rate for encoder prev-step inputs
            "enc_lstm-state_dropout":hp.Fixed(
                "enc_lstm-state_dropout", 0.0),
            ## Activation function for encoder  prev-step input
            "enc_lstm-state_activation":hp.Fixed(
                    "enc_lstm-state_activation", "tanh"),
            ## Initial activation strategy for 'previous output'
            "enc_lstm-state_init":hp.Fixed(
                    "enc_lstm-state_init","glorot_uniform"),

            ## Dropout rate for decoder cell state
            "dec_lstm-cell_dropout":hp.Fixed(
                    "dec_lstm-cell_dropout", 0.0),
            ## Activation function for decoder cell state
            "dec_lstm-cell_activation":hp.Fixed(
                    "dec_lstm-cell_activation", "sigmoid"),
            ## Initial weight activation strategy for encoder cell state
            "dec_lstm-cell_init":hp.Fixed(
                    "dec_lstm-cell_init","orthogonal"),

            ## Dropout rate for decoder inputs
            "dec_lstm-state_dropout":hp.Fixed(
                    "dec_lstm-state_dropout", 0.0),
            ## Activation function for decoder
            "dec_lstm-state_activation":hp.Fixed(
                    "dec_lstm-state_activation", "tanh"),
            ## initial activation for 'previous output'
            "dec_lstm-state_init":hp.Fixed(
                    "dec_lstm-state_init","glorot_uniform"),
            }

    ## Initialize the model with the above arguments.
    model, encoder, decoder = basic_lstmae(
        seq_len=hdict.get("in_seq_len"),
        feat_len=hdict.get("in_feat_len"),
        enc_nodes=hdict.get("enc_lstm_nodes"),
        dec_nodes=hdict.get("dec_lstm_nodes"),
        latent=hdict.get("latent"),
        latent_activation=hdict.get("latent_activation"),
        dropout_rate=hdict.get("dropout_rate"),
        batchnorm=hdict.get("batchnorm"),
        mask_val=hdict.get("mask_val"),
        bidirectional=hdict.get("bidirectional"),
        enc_lstm_kwargs={
            "dropout":hdict.get("enc_lstm-state_dropout"),
            "activation":hdict.get("enc_lstm-state_activation"),
            "kernel_initializer":hdict.get("enc_lstm-state_init"),
            "recurrent_dropout":hdict.get("enc_lstm-cell_dropout"),
            "recurrent_activation":hdict.get("enc_lstm-cell_activation"),
            "recurrent_initializer":hdict.get("enc_lstm-cell_init"),
            },
        dec_lstm_kwargs={
            "dropout":hdict.get("dec_lstm-state_dropout"),
            "activation":hdict.get("dec_lstm-state_activation"),
            "kernel_initializer":hdict.get("dec_lstm-state_init"),
            "recurrent_dropout":hdict.get("dec_lstm-cell_dropout"),
            "recurrent_activation":hdict.get("dec_lstm-cell_activation"),
            "recurrent_initializer":hdict.get("dec_lstm-cell_init"),
            },
        )
    model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hdict.get("learning_rate"),
                ),
            loss="mse",
            metrics=["mse"]
            )
    return model#, enc, dec
'''

if __name__=="__main__":
    buffer_dir = Path("data/swath_sample")

    #swath_dir = Path("data/swath_sample")
    #swath_paths = (p for p in swath_dir.iterdir() if "pkl" in p.name)
    #swaths = (pkl.load(p.open("rb")) for p in swath_paths)

    #ceres_zarr_path = Path("data/buffer/ceres_training.zip")
    #modis_zarr_path = Path("data/buffer/modis_training.zip")
    ceres_zarr_path = Path("/rstor/mdodson/aes770hw4/ceres_validation.zip")
    modis_zarr_path = Path("/rstor/mdodson/aes770hw4/modis_validation.zip")

    """ Load swaths into the zarr arrays """
    #swaths_to_zarr(swaths, ceres_zarr_path, modis_zarr_path)

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
