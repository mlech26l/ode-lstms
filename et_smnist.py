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
from irregular_sampled_datasets import ETSMnistData


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.0005, type=float)
args = parser.parse_args()

data = ETSMnistData(time_major=False)

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
dense_layer = tf.keras.layers.Dense(10)

output_states = rnn((pixel_input, time_input), mask=mask_input)
y = dense_layer(output_states)

model = tf.keras.Model(inputs=[pixel_input, time_input, mask_input], outputs=[y])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.summary()
# Other possibility: use best test accuracy achieved during training
# hist = model.fit(x=(data.train_events,data.train_elapsed,data.train_mask),y=data.train_y,batch_size=128,epochs=args.epochs,validation_data=((data.test_events,data.test_elapsed,data.test_mask),data.test_y))
# test_accuracies = hist.history["val_sparse_categorical_accuracy"]
# best_test_acc = np.max(test_accuracies)

# Fit and evaluate
hist = model.fit(
    x=(data.train_events, data.train_elapsed, data.train_mask),
    y=data.train_y,
    batch_size=128,
    epochs=args.epochs,
)

_, best_test_acc = model.evaluate(
    x=(data.test_events, data.test_elapsed, data.test_mask), y=data.test_y, verbose=2
)

os.makedirs("results/smnist", exist_ok=True)
with open("results/smnist/{}_{}.csv".format(args.model, args.size), "a") as f:
    f.write("{:06f}\n".format(best_test_acc))