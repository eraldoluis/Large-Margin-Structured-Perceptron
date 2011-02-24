package br.pucrio.inf.learn.structlearning.algorithm;

import br.pucrio.inf.learn.structlearning.data.FeatureVector;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.TaskAdapter;

public class Perceptron {

	protected FeatureVector weightVector;
	protected TaskAdapter taskAdapter;
	protected double learningRate;

	public void train(ExampleInput example, ExampleOutput correctOutput,
			FeatureVector correctExampleFeatures) {

		ExampleOutput predictedOutput = taskAdapter.inference(weightVector,
				example);
		FeatureVector predictedExampleFeatures = taskAdapter.extractFeatures(
				example, predictedOutput);

		// Update the
		weightVector.increment(correctExampleFeatures.difference(
				predictedExampleFeatures).scale(learningRate));

	}

}
