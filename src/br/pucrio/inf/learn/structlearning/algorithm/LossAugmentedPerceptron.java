package br.pucrio.inf.learn.structlearning.algorithm;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Inference;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * Perceptron implementation that uses a loss-augmented objective function to
 * update the model weights. It is also aware of partially-labeled examples.
 * 
 * @author eraldof
 * 
 */
public class LossAugmentedPerceptron extends Perceptron {

	/**
	 * This is the loss weight in the objective function for annotated elements.
	 */
	protected double lossAnnotatedWeight;

	/**
	 * This is the loss weight in the objective function for NON-annotated
	 * elements.
	 * 
	 * If this value is less than zero, it is ignored and the weight loss for
	 * annotated elements is used regardless the element be annotated or not.
	 */
	protected double lossNonAnnotatedWeight;

	/**
	 * Increment (per epoch) to the non-annotated elements loss weight
	 */
	protected double lossNonAnnotatedWeightInc;

	public LossAugmentedPerceptron(Inference inferenceImpl, Model initialModel) {
		super(inferenceImpl, initialModel);
		this.lossAnnotatedWeight = 1d;
		this.lossNonAnnotatedWeight = -1d;
		this.lossNonAnnotatedWeightInc = 0d;
	}

	public LossAugmentedPerceptron(Inference inferenceImpl, Model initialModel,
			int numberOfIterations, double learningRate, double lossWeight,
			boolean randomize, boolean averageWeights,
			LearningRateUpdateStrategy learningRateUpdateStrategy) {
		super(inferenceImpl, initialModel, numberOfIterations, learningRate,
				randomize, averageWeights, learningRateUpdateStrategy);
		this.lossAnnotatedWeight = lossWeight;
		this.lossNonAnnotatedWeight = -1d;
		this.lossNonAnnotatedWeightInc = 0d;
	}

	public LossAugmentedPerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight,
			double lossNonAnnotatedWeightInc, boolean randomize,
			boolean averageWeights,
			LearningRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				randomize, averageWeights, learningRateUpdateStrategy);
		this.lossAnnotatedWeight = lossAnnotatedWeight;
		this.lossNonAnnotatedWeight = lossNonAnnotatedWeight;
		this.lossNonAnnotatedWeightInc = lossNonAnnotatedWeightInc;
	}

	@Override
	public double trainOneEpoch(ExampleInput[] inputs, ExampleOutput[] outputs,
			ExampleOutput[] predicteds, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {
		double loss = super.trainOneEpoch(inputs, outputs, predicteds,
				featureEncoding, stateEncoding);
		if (lossNonAnnotatedWeight >= 0d && lossNonAnnotatedWeightInc != 0d)
			// Increment the loss weight for non-annotated elements.
			lossNonAnnotatedWeight = Math.min(lossAnnotatedWeight,
					lossNonAnnotatedWeight + lossNonAnnotatedWeightInc);
		return loss;
	}

	@Override
	public double trainOneEpoch(ExampleInput[] inputsA,
			ExampleOutput[] outputsA, ExampleOutput[] predictedsA,
			double weightA, ExampleInput[] inputsB, ExampleOutput[] outputsB,
			ExampleOutput[] predictedsB, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {
		double loss = super.trainOneEpoch(inputsA, outputsA, predictedsA,
				weightA, inputsB, outputsB, predictedsB, featureEncoding,
				stateEncoding);
		if (lossNonAnnotatedWeight >= 0d && lossNonAnnotatedWeightInc != 0d)
			// Increment the loss weight for non-annotated elements.
			lossNonAnnotatedWeight = Math.min(lossAnnotatedWeight,
					lossNonAnnotatedWeight + lossNonAnnotatedWeightInc);
		return loss;
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

		if (lossNonAnnotatedWeight < 0)
			// Predict the best output structure to the current input structure
			// using a loss-augmented objective function.
			inferenceImpl.lossAugmentedInference(model, input, referenceOutput,
					predictedOutput, lossAnnotatedWeight);
		else
			// Predict the best output structure to the current input structure
			// using a loss-augmented objective function that uses different
			// weights for annotated and non-annotated tokens.
			inferenceImpl.lossAugmentedInference(model, input, correctOutput,
					referenceOutput, predictedOutput, lossAnnotatedWeight,
					lossNonAnnotatedWeight);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, referenceOutput, predictedOutput,
				getCurrentLearningRate());

		// Debug.
		if (DebugUtil.print && loss != 0d)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) referenceOutput,
					(SequenceOutput) predictedOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged
		// weights.
		model.sumUpdates(iteration);

		return loss;

	}
}
