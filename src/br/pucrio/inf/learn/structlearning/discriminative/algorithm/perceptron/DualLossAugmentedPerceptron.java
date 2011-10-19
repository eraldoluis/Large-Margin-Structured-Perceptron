package br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.task.DualModel;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * Dual (kernelized) perceptron algorithm that uses a loss-augmented strategy.
 * 
 * The algorithm receives an inference algorithm and a dual model. The dual
 * model parameters are based on an array of examples and it uses the examples
 * indexes within this array to reference each example. So, the array of
 * training examples given to the train method of this class needs to be the
 * same (or follow the same order) of the array given to the model.
 * 
 * @author eraldo
 * 
 */
public class DualLossAugmentedPerceptron extends LossAugmentedPerceptron {

	/**
	 * Indicate whether the distillation procedure is turned on or not.
	 */
	private boolean distill;

	/**
	 * Just a cast to the current model.
	 */
	private DualModel dualModel;

	public DualLossAugmentedPerceptron(Inference inferenceImpl,
			DualModel initialModel) {
		super(inferenceImpl, initialModel);
	}

	public DualLossAugmentedPerceptron(Inference inferenceImpl,
			DualModel initialModel, int numberOfIterations,
			double learningRate, double lossWeight, boolean randomize,
			boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		super(inferenceImpl, initialModel, numberOfIterations, learningRate,
				lossWeight, randomize, averageWeights,
				learningRateUpdateStrategy);
		dualModel = (DualModel) model;
	}

	public DualLossAugmentedPerceptron(Inference taskImpl,
			DualModel initialModel, int numberOfIterations,
			double learningRate, double lossAnnotatedWeight,
			double lossNonAnnotatedWeight, double lossNonAnnotatedWeightInc,
			boolean randomize, boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				lossAnnotatedWeight, lossNonAnnotatedWeight,
				lossNonAnnotatedWeightInc, randomize, averageWeights,
				learningRateUpdateStrategy);
		dualModel = (DualModel) model;
	}

	@Override
	public void train(ExampleInput[] inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInput[] inputsB,
			ExampleOutput[] outputsB, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		/*
		 * Need to adequate the algorithm to store a unique array of all
		 * examples or modify the <code>DualHmm</code> class to deal with more
		 * than one examples arrays.
		 */
		throw new NotImplementedException();
	}

	@Override
	public double train(ExampleInput input, ExampleOutput correctOutput,
			ExampleOutput predictedOutput) {

		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			/*
			 * If the user asked to consider partially-labeled examples then
			 * infer the missing values within the given correct output
			 * structure before updating the current model.
			 */
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
					predictedOutput, lossAnnotatedWeight);
		else
			/*
			 * Predict the best output structure to the current input structure
			 * using a loss-augmented objective function that uses different
			 * weights for annotated and non-annotated tokens.
			 */
			inferenceImpl.lossAugmentedInferenceWithPartiallyLabeledReference(
					model, input, correctOutput, referenceOutput,
					predictedOutput, lossAnnotatedWeight,
					lossNonAnnotatedWeight);

		// Update the current model and return the loss for this example.
		double loss = dualModel.update(indexCurrentExample, referenceOutput,
				predictedOutput, getCurrentLearningRate());

		if (distill && loss > 0) //((iteration + 1) % 10 == 0))
			dualModel.distill(inferenceImpl, lossAnnotatedWeight, predicteds);

		// Debug.
		if (DebugUtil.print)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) referenceOutput,
					(SequenceOutput) predictedOutput, loss);

		/*
		 * Averaged-Perceptron: account the updates into the averaged weights.
		 */
		model.sumUpdates(iteration);

		++iteration;

		return loss;
	}

	/**
	 * Return whether the distillation process is activated.
	 * 
	 * @return
	 */
	public boolean isDistill() {
		return distill;
	}

	/**
	 * Activate or deactivate the distallation processo for the next training
	 * procedure.
	 * 
	 * @param distill
	 */
	public void setDistill(boolean distill) {
		this.distill = distill;
	}

}
