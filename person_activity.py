# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import tensorflow as tf
import argparse
from node_cell import (
    LSTMCell,
    CTRNNCell,
    ODELSTM,
    VanillaRNN,
    CTGRU,
    BidirectionalRNN,
    GRUD,
    PhasedLSTM,
    GRUODE,
    HawkLSTMCell,
)
from irregular_sampled_datasets import PersonData

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--dense", action="store_true")
args = parser.parse_args()

data = PersonData()

if args.model == "lstm":
    cell = LSTMCell(units=args.size)
elif args.model == "ctrnn":
    cell = CTRNNCell(units=args.size, num_unfolds=3, method="rk4")
elif args.model == "node":
    cell = CTRNNCell(units=args.size, num_unfolds=3, method="rk4", tau=0)
elif args.model == "odelstm":
    cell = ODELSTM(units=args.size)
elif args.model == "ctgru":
    cell = CTGRU(units=args.size)
elif args.model == "vanilla":
    cell = VanillaRNN(units=args.size)
elif args.model == "bidirect":
    cell = BidirectionalRNN(units=args.size)
elif args.model == "grud":
    cell = GRUD(units=args.size)
elif args.model == "phased":
    cell = PhasedLSTM(units=args.size)
elif args.model == "gruode":
    cell = GRUODE(units=args.size)
elif args.model == "hawk":
    cell = HawkLSTMCell(units=args.size)
else:
    raise ValueError("Unknown model type '{}'".format(args.model))

pixel_input = tf.keras.Input(shape=(data.seq_len, data.feature_size), name="features")
time_input = tf.keras.Input(shape=(data.seq_len, 1), name="time")

rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)
dense_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(data.num_classes))

output_states = rnn((pixel_input, time_input))
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input, time_input], outputs=[y])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()

# Fit and evaluate
hist = model.fit(
    x=(data.train_x, data.train_t),
    y=data.train_y,
    batch_size=128,
    epochs=args.epochs,
    validation_data=((data.test_x, data.test_t), data.test_y),
)
_, best_test_acc = model.evaluate(
    x=(data.test_x, data.test_t), y=data.test_y, verbose=2
)

# log results
base_path = "results/person_activity"
os.makedirs(base_path, exist_ok=True)
with open("{}/{}_{}.csv".format(base_path, args.model, args.size), "a") as f:
    f.write("{:06f}\n".format(best_test_acc))