# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.

import os
import torch
import argparse
from irregular_sampled_datasets import PersonData, ETSMnistData, XORData
import torch.utils.data as data
from torch_node_cell import ODELSTM, IrregularSequenceLearner
import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="person")
parser.add_argument("--solver", default="dopri5")
parser.add_argument("--size", default=64, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--gpus", default=0, type=int)
args = parser.parse_args()


def load_dataset(args):
    if args.dataset == "person":
        dataset = PersonData()
        train_x = torch.Tensor(dataset.train_x)
        train_y = torch.LongTensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_t)
        test_x = torch.Tensor(dataset.test_x)
        test_y = torch.LongTensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_t)
        train = data.TensorDataset(train_x, train_ts, train_y)
        test = data.TensorDataset(test_x, test_ts, test_y)
        return_sequences = True
    else:
        if args.dataset == "et_mnist":
            dataset = ETSMnistData(time_major=False)
        elif args.dataset == "xor":
            dataset = XORData(time_major=False, event_based=True, pad_size=32)
        else:
            raise ValueError("Unknown dataset '{}'".format(args.dataset))
        return_sequences = False
        train_x = torch.Tensor(dataset.train_events)
        train_y = torch.LongTensor(dataset.train_y)
        train_ts = torch.Tensor(dataset.train_elapsed)
        train_mask = torch.Tensor(dataset.train_mask)
        test_x = torch.Tensor(dataset.test_events)
        test_y = torch.LongTensor(dataset.test_y)
        test_ts = torch.Tensor(dataset.test_elapsed)
        test_mask = torch.Tensor(dataset.test_mask)
        train = data.TensorDataset(train_x, train_ts, train_y, train_mask)
        test = data.TensorDataset(test_x, test_ts, test_y, test_mask)
    trainloader = data.DataLoader(train, batch_size=64, shuffle=True, num_workers=4)
    testloader = data.DataLoader(test, batch_size=64, shuffle=False, num_workers=4)
    in_features = train_x.size(-1)
    num_classes = int(torch.max(train_y).item() + 1)
    return trainloader, testloader, in_features, num_classes, return_sequences


trainloader, testloader, in_features, num_classes, return_sequences = load_dataset(args)

ode_lstm = ODELSTM(
    in_features,
    args.size,
    num_classes,
    return_sequences=return_sequences,
    solver_type=args.solver,
)
learn = IrregularSequenceLearner(ode_lstm, lr=args.lr)
trainer = pl.Trainer(
    max_epochs=args.epochs,
    progress_bar_refresh_rate=1,
    gradient_clip_val=1,
    gpus=args.gpus,
)
trainer.fit(learn, trainloader)

results = trainer.test(learn, testloader)
base_path = "results/{}".format(args.dataset)
os.makedirs(base_path, exist_ok=True)
with open("{}/pt_ode_lstm_{}.csv".format(base_path, args.size), "a") as f:
    f.write("{:06f}\n".format(results[0]["val_acc"]))