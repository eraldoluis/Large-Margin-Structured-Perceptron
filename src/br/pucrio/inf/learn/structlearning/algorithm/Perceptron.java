package br.pucrio.inf.learn.structlearning.algorithm;

import java.util.Iterator;
import java.util.LinkedList;

import br.pucrio.inf.learn.structlearning.data.FeatureVector;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.TaskAdapter;

public class Perceptron {

	protected FeatureVector weightVector;
	protected TaskAdapter taskAdapter;
	protected double learningRate;
	protected int numberOfIterations;

	public Perceptron(TaskAdapter taskAdapter) {
		this.taskAdapter = taskAdapter;
		weightVector = new FeatureVector();
		learningRate = 1d; // Collins' original learning rate
	}

	public void train(Iterable<ExampleInput> exampleInputs,
			Iterable<ExampleOutput> exampleOutputs) {

		LinkedList<FeatureVector> correctExamplesAsFeatures = new LinkedList<FeatureVector>();

		// Create feature representations of each input-output example.
		Iterator<ExampleInput> inIt = exampleInputs.iterator();
		Iterator<ExampleOutput> outIt = exampleOutputs.iterator();
		while (inIt.hasNext() && outIt.hasNext()) {
			correctExamplesAsFeatures.add(taskAdapter.extractFeatures(inIt
					.next(), outIt.next()));
		}

		for (int iter = 0; iter < numberOfIterations; ++iter) {
			// Iterate over the training examples, updating the weight vector.
			inIt = exampleInputs.iterator();
			outIt = exampleOutputs.iterator();
			Iterator<FeatureVector> ftrsIt = correctExamplesAsFeatures
					.iterator();
			while (inIt.hasNext() && outIt.hasNext() && ftrsIt.hasNext())
				train(inIt.next(), outIt.next(), ftrsIt.next());
		}
	}

	public void train(ExampleInput example, ExampleOutput correctOutput,
			FeatureVector correctExampleFeatures) {

		ExampleOutput predictedOutput = taskAdapter.inference(weightVector,
				example);
		FeatureVector predictedExampleFeatures = taskAdapter.extractFeatures(
				example, predictedOutput);

		// Update the current weight vector.
		weightVector.increment(correctExampleFeatures.difference(
				predictedExampleFeatures).scale(learningRate));

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

}
