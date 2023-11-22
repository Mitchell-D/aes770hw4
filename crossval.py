from pathlib import Path
from random import random
import pickle as pkl

import zarr
import numpy as np
import os

import keras_tuner
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

    c_early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4)
    c_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss", save_best_only=True,
            filepath=buffer_dir.joinpath(
                "lstmae_0_{epoch}_{mse:.2f}.hdf5"))
    c_csvlog = tf.keras.callbacks.CSVLogger(
            buffer_dir.joinpath("lstmae_0_prog.csv"))

    """ Use hyperparameter tuning """
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
