# Learning Long-Term Dependencies in Irregularly-Sampled Time Series

This is the official code repository of the paper [Learning Long-Term Dependencies in Irregularly-Sampled Time Series](https://arxiv.org/pdf/2006.04418.pdf).

The principle idea of this paper is to combine ODE-RNNs with LSTM in a single architecture that 

- has the supreme modeling power of ordinary differential equations for fitting irregularly-sampled time series, and
- the capability of the LSTM architecture for learning long-term dependencies in the data.

Our proposed ODE-LSTM achieves this by post-processing the output-state of the LSTM's state-pair by a learnable ODE.  The LSTM's memory-state stays untouched to allow a near-constant backpropagation of the error signal. 

![alt](misc/state_table.png)

## Requirements

**Packages**
- Python 3.5 or newer
- TensorFlow 2.0 or newer

Tested with python3.6/python3.5 and TensorFlow 2.1 on Ubuntu 18.04 and Ubuntu 16.04

## Data preparation

Data for the XOR experiment are generated *on the fly*, i.e., no manual downloading necessary.
The MNIST data are loaded through the ```tf.keras.datasets``` API, i.e., no manual downloading necessary.
Data for the Walker kinematic and the person activity task however, must be downloaded first. 
This can be done by 

```bash
source download_datasets.sh
```


## Module description

- ```node_cell.py```: Implementation of all continuous-time RNNs used in the experimental evaluation in the paper
- ```xor_task.py```: Executable to run the synthetic XOR experiment (both the dense and the event-based modes)
- ```person_activity.py```: Executable to run the Person activity experiment
- ```et_mnist.py```: Executable to run the Event-based sequential MNIST experiment
- ```walker_kinematic.py```: Executable to run the Walker2d kinematic simulation experiment

Each of the four executable python scripts contain the code for loading and pre-processing the data, as well the code to train and evaluate the models.

## Example usage

The four executable python scripts use some command line argument parsing to specify the RNN type and hyperparameters.
The RNN type can be specified by ```--model RNN```, where ```RNN``` is one of the following

| ```--model RNN``` | Description                           |
| ----------------- | ------------------------------------- |
| ```lstm```        | Augmented LSTM                        |
| ```ctrnn```       | CT-RNN                                |
| ```node```        | ODE-RNN                               |
| ```ctgru```       | CT-GRU                                |
| ```grud```        | GRU-D                                 |
| ```gruode```      | GRU-ODE                               |
| ```vanilla```     | Vanilla RNN with time-dependent decay |
| ```bidirect```    | Bidirectional RNN (LSTM with ODE-RNN) |
| ```phased```      | PhasedLSTM                            |
| ```hawk```        | Hawkes process LSTM                   |
| ```odelstm```     | ODE-LSTM (ours)                       |


For instance

```bash
python3 xor_task.py --model lstm --epochs 500 --dense
```

runs the XOR sequence classification experiment with the dense encoding (=regularly sampled time-series).
By omitting the ```--dense``` flag one can run the same experiment but with the event-based encoding (=irregularly sampled time-series)

## Logging

Each executable python script stores the result of the experiment in the directory ```results``` (which will be created if it does not exists).
The ```results``` directory will have the following structure:

- ```results/xor_event``` Results of the event-based XOR task
- ```results/xor_dense``` Results of the dense encoded XOR task
- ```results/smnist``` Results of the event-based sequential MNIST experiment
- ```results/person_activity``` Results of the Person activity dataset
- ```results/walker``` Results of the Walker2d kinematic simulation dataset

The results for different RNN types will be logged in separate files.
For instance, ```results/xor_event/lstm_64.csv``` will contain the results of the augmented LSTM with 64 hidden units on the event-based XOR task. The naming of the RNN models is the same as for the ```--model``` argument as described above.

## Citation

```bibtex
@article{lechner2020learning,
  title={Learning Long-Term Dependencies in Irregularly-Sampled Time Series},
  author={Lechner, Mathias and Hasani, Ramin},
  journal={arXiv preprint arXiv:2006.04418},
  year={2020}
}
```
