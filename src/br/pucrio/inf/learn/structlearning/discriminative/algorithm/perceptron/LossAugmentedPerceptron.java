package br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Perceptron implementation that uses a loss-augmented objective function to
 * update the model weights. It is also aware of partially-labeled examples.
 * 
 * @author eraldof
 * 
 */
public class LossAugmentedPerceptron extends Perceptron {

	/**
	 * This is the loss weight in the objective function.
	 */
	protected double lossWeight;

	/**
	 * Increment in the loss weight after each epoch.
	 */
	protected double lossWeightInc;

	/**
	 * This is the loss weight in the objective function for NON-annotated
	 * elements.
	 * 
	 * If this value is less than zero, it is ignored and the weight loss for
	 * annotated elements is used regardless the element be annotated or not.
	 */
	protected double lossNonAnnotatedWeight;

	/**
	 * Increment (per epoch) to the non-annotated loss weight
	 */
	protected double lossNonAnnotatedWeightInc;

	public LossAugmentedPerceptron(Inference inferenceImpl, Model initialModel) {
		super(inferenceImpl, initialModel);
		this.lossWeight = 1d;
		this.lossNonAnnotatedWeight = -1d;
		this.lossNonAnnotatedWeightInc = 0d;
	}

	public LossAugmentedPerceptron(Inference inferenceImpl, Model initialModel,
			int numberOfIterations, double learningRate, double lossWeight,
			boolean randomize, boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		super(inferenceImpl, initialModel, numberOfIterations, learningRate,
				randomize, averageWeights, learningRateUpdateStrategy);
		this.lossWeight = lossWeight;
		this.lossNonAnnotatedWeight = -1d;
		this.lossNonAnnotatedWeightInc = 0d;
	}

	public LossAugmentedPerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight,
			double lossNonAnnotatedWeightInc, boolean randomize,
			boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				randomize, averageWeights, learningRateUpdateStrategy);
		this.lossWeight = lossAnnotatedWeight;
		this.lossNonAnnotatedWeight = lossNonAnnotatedWeight;
		this.lossNonAnnotatedWeightInc = lossNonAnnotatedWeightInc;
	}

	@Override
	public double trainOneEpoch(ExampleInput[] inputs, ExampleOutput[] outputs,
			ExampleOutput[] predicteds) {
		double loss = super.trainOneEpoch(inputs, outputs, predicteds);
		if (lossNonAnnotatedWeight >= 0d && lossNonAnnotatedWeightInc != 0d)
			// Increment the loss weight for non-annotated elements.
			lossNonAnnotatedWeight = Math.min(lossWeight,
					lossNonAnnotatedWeight + lossNonAnnotatedWeightInc);
		return loss;
	}

	@Override
	public double trainOneEpoch(ExampleInput[] inputsA,
			ExampleOutput[] outputsA, ExampleOutput[] predictedsA,
			double weightA, ExampleInput[] inputsB, ExampleOutput[] outputsB,
			ExampleOutput[] predictedsB) {
		double loss = super.trainOneEpoch(inputsA, outputsA, predictedsA,
				weightA, inputsB, outputsB, predictedsB);
		// Increment the loss weight.
		if (lossWeightInc != 0d) {
			lossWeight += lossWeightInc;
			if (lossWeight < 0d)
				lossWeight = 0d;
		}
		// Increment the loss weight for non-annotated elements.
		if (lossNonAnnotatedWeight >= 0d && lossNonAnnotatedWeightInc != 0d)
			lossNonAnnotatedWeight = Math.min(lossWeight,
					lossNonAnnotatedWeight + lossNonAnnotatedWeightInc);
		return loss;
	}

	@Override
	public double train(ExampleInput input, ExampleOutput correctOutput,
			ExampleOutput predictedOutput) {
		/*
		 * If the user asked to consider partially-labeled examples, then infer
		 * the missing values within the given correct output structure before
		 * updating the current model. Otherwise, the reference (for
		 * loss-function and update) will be the given output structure that
		 * must be completely labeled.
		 */
		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			referenceOutput = correctOutput.createNewObject();
			inferenceImpl.partialInference(model, input, correctOutput,
					referenceOutput);
		}

		if (lossNonAnnotatedWeight < 0)
			/*
			 * Predict the best output structure to the current input structure
			 * using a loss-augmented objective function.
			 */
			inferenceImpl.lossAugmentedInference(model, input, referenceOutput,
					predictedOutput, lossWeight);
		else
			/*
			 * Predict the best output structure to the current input structure
			 * using a loss-augmented objective function that uses different
			 * weights for annotated and non-annotated tokens.
			 */
			inferenceImpl.lossAugmentedInferenceWithPartiallyLabeledReference(
					model, input, correctOutput, referenceOutput,
					predictedOutput, lossWeight, lossNonAnnotatedWeight);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, referenceOutput, predictedOutput,
				getCurrentLearningRate());

		// Averaged perceptron: account the updates into the averaged weights.
		model.sumUpdates(iteration);

		++iteration;

		return loss;
	}

	/**
	 * Set the value to be incremented in the loss weight after each epoch.
	 * 
	 * @param increment
	 */
	public void setLossWeightIncrement(double increment) {
		lossWeightInc = increment;
	}
}
