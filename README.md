# AI Driving Olympics

<a href="http://aido.duckietown.org"><img width="200" src="https://www.duckietown.org/wp-content/uploads/2018/07/AIDO-768x512.png"/></a>


## "Imitation Learning from Simulation" for challenge `aido1_LF1-v3`

This is a behavioral cloning (the simplest IL implementation) baseline for the lane following challenge.

The [online description of this challenge is here][online].

For submitting, please follow [the instructions available in the book][book].

[book]: http://docs.duckietown.org/DT18/AIDO/out/

[online]: https://challenges.duckietown.org/v3/humans/challenges/aido1_LF1-v3

## Description

This repository contains a baseline implementation of Imiation Learning via Supervised Learning (a.k.a Behavioral Cloning).
This is probably the simplest way to address this problem.
To achieve our goal to directly estimate a policy for the Lane Following problem, we divided this baseline into three parts:
Logging, Training, and Evaluation.
We leave it open to your creativity any other implementations that can be derived from baseline.

## Step 1: Logging

Most of of the logging procedure is implemented on `log.py` and `_loggers.py`.
There are two crucial aspects that can impact your final results:

1. The quality of the expert.
2. The number and variety of samples.

The performance of pure pursuit controller implemented on `teacher.py` is not precisely great.
Even though it uses the internal state of the environment to compute the appropriate action, there are several parameters that need to be fine tuned.
We have prepare some basic debugging capabilities (the `DEBUG=False` flag, line HERE) to help you debug this implementation.
In any case, feel free to provide an expert implementation of your own creation.

Another important aspect you need to take care of is the number of samples.
The number of samples logged are controlled by the `EPISODES` and `STEPS` parameters.
Obviously, the bigger these numbers are, the more samples you get.

As we are using Deep Learning here, the amount of data is crucial, but so it is the variety of the samples we see.
Remember, we are estimating a policy, so the better we capture the underlying distribution of the data, the more robust our policy is.


## Step 2: Training

The output of the logging procedure is a file that we called `train.log`, but you can rename it to your convenience.
We have prepared a very simple `Reader` class in `_loggers.py` that is capable of reading the logs we store in the previous step.

The training procedure implemented in `train.py` is quite simple.
The baseline CNN is a one Residual module (Resnet1) network trained to regress the velocity and the steering angle of our simulated duckiebot.
All the Tensorflow boilerplate code is encapsulated in `TensorflowModel` class implemented on `model.py`.
You may find this abstraction quite useful as it is already handling model initialization and persistence ofr you.

Then, the training procedure is quite clear.
We trained the model for a number of `EPOCHS`, using `BATCH_SIZE` samples at each step and the model is persisted every 10 episodes of training.


## Step 3 Evaluation
A simple evaluation script `eval.py` is provided with this implementation.
It loads the latest model from the `trained_models` directory and runs it on the simulator.
Although it is not an official metric for the challenge, you can use the cumulative reward emitted by the `gym` to evaluate the performance of your latest model.

With the current implementation and hyper-parameters selection, we get something like:

```
total reward: -5646.22255589, mean reward: -565.0
```

which is not by any standard a good performance.

# Submitting


