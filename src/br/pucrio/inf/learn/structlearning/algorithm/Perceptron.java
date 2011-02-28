package br.pucrio.inf.learn.structlearning.algorithm;

import java.util.Iterator;
import java.util.LinkedList;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.Model;

/**
 * Perceptron-trained linear classifier with structural examples.
 * 
 * @author eraldof
 * 
 */
public class Perceptron {

	/**
	 * Task-specific model.
	 */
	protected Model model;

	/**
	 * Learning rate.
	 */
	protected double learningRate;

	/**
	 * Number of iterations.
	 */
	protected int numberOfIterations;

	/**
	 * Create a perceptron to train the given initial model using the default
	 * Collins' learning rate (1) and the default number of iterations (10).
	 * 
	 * @param initialModel
	 */
	public Perceptron(Model initialModel) {
		this(initialModel, 10, 1d);
	}

	/**
	 * Create a perceptron to train the given initial model using the given
	 * number of iterations and learning rate.
	 * 
	 * @param initialModel
	 * @param numberOfIterations
	 * @param learningRate
	 */
	public Perceptron(Model initialModel, int numberOfIterations,
			double learningRate) {
		this.model = initialModel;
		this.numberOfIterations = numberOfIterations;
		this.learningRate = learningRate;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public int getNumberOfIterations() {
		return numberOfIterations;
	}

	public void setNumberOfIterations(int numberOfIterations) {
		this.numberOfIterations = numberOfIterations;
	}

	/**
	 * Train the model with the given examples, iterating according to the
	 * number of iterations. Corresponding inputs and outputs must be in the
	 * same order.
	 * 
	 * @param inputs
	 * @param outputs
	 */
	public void train(Iterable<ExampleInput> inputs,
			Iterable<ExampleOutput> outputs) {

		// Iterators for inputs and outputs (correct and predicted).
		Iterator<ExampleInput> itIn;
		Iterator<ExampleOutput> itOut, itPred;

		// Create a predicted output object for each example.
		LinkedList<ExampleOutput> predicteds = new LinkedList<ExampleOutput>();
		itOut = outputs.iterator();
		while (itOut.hasNext())
			predicteds.add(itOut.next().createNewObject());

		for (int iter = 0; iter < numberOfIterations; ++iter) {
			// Iterate over the training examples, updating the weight vector.
			itIn = inputs.iterator();
			itOut = outputs.iterator();
			itPred = predicteds.iterator();
			while (itIn.hasNext() && itOut.hasNext() && itPred.hasNext())
				train(itIn.next(), itOut.next(), itPred.next());
		}
	}

	/**
	 * Update the current model using the two given outputs for one input.
	 * 
	 * @param input
	 * @param correctOutput
	 * @param predictedOutput
	 */
	public void train(ExampleInput input, ExampleOutput correctOutput,
			ExampleOutput predictedOutput) {

		// Predict the best output with the current mobel.
		model.inference(input, predictedOutput);

		// Update the current model.
		model.update(input, correctOutput, predictedOutput, learningRate);

	}

}
