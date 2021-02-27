
# Largin Margin Structured Perceptron

Author: Eraldo R. Fernandes
(C) 2011

## What is structured learning?

Structured learning consists in learning a mapping from inputs to structured
outputs by means of a sample of correct input-output pairs. Many important
problems fit in this setting. For instance, dependency parsing involves the
recognition of a tree underlying a sentence.


## What is structured perceptron?

Structured perceptron is a training algorithm for structured problems that is a generalization of the binary perceptron.
It learns the parameters of a linear discriminant function that, given an input,
  discriminates the correct output structure from the alternative ones by means of a task-specific optimization problem
    (an inference algorithm).


## What is included in this framework?

There are implementations of different variations of the structured perceptron algorithm,
  namely:
* vanilla,
* large margin, and
* dual (kernelized).

The are also some instantiations of the framework for different tasks:
* sequence labeling (POS tagging, text chunking, etc.),
* dependency parsing,
* quotation extraction, and
* coreference resolution.


## How to execute an instance of the framework?

The common entry point for all instantiations is the command

  java [br.pucrio.inf.learn.structlearning.discriminative.driver.Driver](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/driver/Driver.java)

When executed, this command shows a list available sub-commands.
For instance, you have the sub-command TrainDP to train and evaluate dependency parsing models.
You can execute each sub-command without arguments to access their list of options.


## Acknowledgement

This implementation has been and is conducted through the help of several colaborators.
Specially, I would like to thank Ulf Brefeld and Ruy L. Milidiú.
Both have given many relevant advices and ideas during the development of this framework.
