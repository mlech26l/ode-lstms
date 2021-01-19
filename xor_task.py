# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import tensorflow as tf
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
import argparse
from irregular_sampled_datasets import XORData


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument("--dense", action="store_true")
args = parser.parse_args()

if args.dense:
    data = XORData(time_major=False, event_based=False, pad_size=32)
else:
    data = XORData(time_major=False, event_based=True, pad_size=32)

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

pixel_input = tf.keras.Input(shape=(data.pad_size, 1), name="pixel")
time_input = tf.keras.Input(shape=(data.pad_size, 1), name="time")
mask_input = tf.keras.Input(shape=(data.pad_size,), dtype=tf.bool, name="mask")

rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
dense_layer = tf.keras.layers.Dense(1)

output_states = rnn((pixel_input, time_input), mask=mask_input)
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[y])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],
)
model.summary()

# Fit model
hist = model.fit(
    x=(data.train_events, data.train_elapsed, data.train_mask),
    y=data.train_y,
    batch_size=256,
    epochs=args.epochs,
)
# Evaluate model after training
_, best_test_acc = model.evaluate(
    x=(data.test_events, data.test_elapsed, data.test_mask), y=data.test_y, verbose=2
)

# log results
if args.dense:
    base_path = "results/xor_dense"
else:
    base_path = "results/xor_event"
os.makedirs(base_path, exist_ok=True)
with open("{}/{}_{}.csv".format(base_path, args.model, args.size), "a") as f:
    f.write("{:06f}\n".format(best_test_acc))