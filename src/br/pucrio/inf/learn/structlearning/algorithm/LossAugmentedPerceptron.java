package br.pucrio.inf.learn.structlearning.algorithm;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.Inference;
import br.pucrio.inf.learn.structlearning.task.Model;

/**
 * Perceptron implementation that uses a loss-augmented objective function to
 * update the model weights. It is also aware of partially-labeled examples.
 * 
 * @author eraldof
 * 
 */
public class LossAugmentedPerceptron extends Perceptron {

	/**
	 * This is the weight of the loss term in the objective function.
	 */
	protected double lossWeight;

	public LossAugmentedPerceptron(Inference taskImpl, Model initialModel) {
		super(taskImpl, initialModel);
		this.lossWeight = 1d;
	}

	public LossAugmentedPerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate, double lossWeight) {
		super(taskImpl, initialModel, numberOfIterations, learningRate);
		this.lossWeight = lossWeight;
	}

	@Override
	public double trainOneExample(ExampleInput input,
			ExampleOutput correctOutput, ExampleOutput predictedOutput) {

		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			// If the user asked to consider partially-labeled examples then
			// infer the missing values within the given correct output
			// structure before updating the current model.
			referenceOutput = correctOutput.createNewObject();
			inferenceImpl.partialInference(model, input, correctOutput,
					referenceOutput);
		}

		// Predict the best output structure to the current input structure
		// using a loss-augmented objective function.
		inferenceImpl.lossAugmentedInference(model, input, referenceOutput,
				predictedOutput, lossWeight);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, referenceOutput, predictedOutput,
				learningRate);

		// TODO debug
		// if (DebugUtil.print && loss != 0d)
		// DebugUtil.printSequence((SequenceInput) input,
		// (SequenceOutput) correctOutput,
		// (SequenceOutput) referenceOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged
		// weights.
		model.sumUpdates(iteration);

		return loss;

	}
}
