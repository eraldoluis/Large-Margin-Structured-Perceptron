# Framework Overview
In the following, we highlight the most important aspects of this framework,
  including the *hot spots*.


## Main classes

In the diagram below, we present the class `Perceptron`, which takes care of the training loop,
  along with the most important interfaces, which correspond to the hot spots of this framework.

![LMSP-train-classes](https://user-images.githubusercontent.com/708031/109401384-b6f19000-7924-11eb-9a1c-2e603197b2d1.png)

The main responsabilities of each class and interface are:
* [**`Perceptron`**](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/algorithm/perceptron/Perceptron.java) &nbsp;&nbsp; 
  This class implements the vanilla Structured Perceptron training algorithm.
  It also serves as base class for other variants of the Structured Perceptron (large margin and dual, for instance).
  This implementation should works for any instantiation of the framework.
* [**`Model`**](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/task/Model.java) &nbsp;&nbsp; 
  The class that implements this interface must store the model parameters and implement some methods.
  The most important method is:
  ```java
  double update(ExampleInput in, ExampleOutput out, ExampleOutput pred, double lr);
  ```
  This method gets an input `in`, its correct output `out`, the current prediction `pred`, the learning rate `lr` and
    must update the model parameters according to the difference between `out` and `pred`.
* [**`Inference`**](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/task/Inference.java) &nbsp;&nbsp; 
  This interface represents the inference algorithm, i.e., the algorithm that predicts an output structure for a given input and model.
  The inference method is:
  ```java
  void inference(Model model, ExampleInput input, ExampleOutput output);
  ```
* [**`Dataset`**](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/data/Dataset.java) &nbsp;&nbsp; 
  This interface must be implemented to provide training examples,
    usualy by reading them from a file.
* [**`ExampleInput`**](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/data/ExampleInput.java) &nbsp;&nbsp; 
  This interface represents the input features of an example.
* [**`ExampleOutput`**](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/blob/master/src/br/pucrio/inf/learn/structlearning/discriminative/data/ExampleOutput.java) &nbsp;&nbsp; 
  This interface represents the output structure of an example.

## Training loop

In the following sequence diagram, we present an overview of the operations within a typical training loop.

![LMSP-train-sequence](https://user-images.githubusercontent.com/708031/109402301-2cf8f580-792b-11eb-8b79-b718e155385b.png)

The client code (the training script) must create the `Dataset` object and obtain the list of inputs and outputs.
It then passes those to the training algorithm (a `Perceptron` object, for example)
  that will execute the training loop.

In a typical iteration of the training loop, the algorithm will
  randomly select a training example (the pair `in`, `out`),
  call the `inference(...)` method of the `Inference` object to obtain a predicted structure `pred`,
  and finally update the model weights considering the difference between the correct structure `out` and the predicted one `pred`.


## Framework Instantiations

If you want to learn about a specific instantiation of the framework,
  you can have a look at the sub-packages of the package [`application`](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/tree/master/src/br/pucrio/inf/learn/structlearning/discriminative/application).
And, in the package [`driver`](https://github.com/eraldoluis/Large-Margin-Structured-Perceptron/tree/master/src/br/pucrio/inf/learn/structlearning/discriminative/driver)
  you can find the training scripts for each instantiation.
