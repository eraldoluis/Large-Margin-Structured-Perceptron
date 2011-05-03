package br.pucrio.inf.learn.structlearning.algorithm.perceptron;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.Inference;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * McAllester et al.'s Perceptron implementation that uses a modified updating
 * rule which is proved to directly optimize the used loss function.
 * 
 * The difference from the toward-better implementation (this one) to the
 * away-from-worse implementation is that the former uses a loss-augmented
 * inference that privileges low-loss solutions and update the model weights
 * toward the loss-augmented solution that is better than the non-loss-augmented
 * solution.
 * 
 * On the other hand, the away-from-worse implementation privileges high-loss
 * solutions and update the model weights away from this worse solution.
 * 
 * @author eraldof
 * 
 */
public class TowardBetterPerceptron extends LossAugmentedPerceptron {

	public TowardBetterPerceptron(Inference taskImpl, Model initialModel) {
		super(taskImpl, initialModel);
	}

	public TowardBetterPerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate, double lossWeight,
			boolean randomize, boolean averageWeights,
			LearningRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				lossWeight, randomize, averageWeights,
				learningRateUpdateStrategy);
	}

	public TowardBetterPerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight,
			double lossNonAnnotatedWeightInc, boolean randomize,
			boolean averageWeights,
			LearningRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				lossAnnotatedWeight, lossNonAnnotatedWeight,
				lossNonAnnotatedWeightInc, randomize, averageWeights,
				learningRateUpdateStrategy);
	}

	@Override
	public double trainOneExample(ExampleInput input,
			ExampleOutput correctOutput, ExampleOutput predictedOutput) {

		// Infer the whole output structure. This is the "worse" output
		// structure used to update the model.
		inferenceImpl.inference(model, input, predictedOutput);

		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			// If the user asked to consider partially-labeled examples then
			// infer the missing values within the given correct output
			// structure before updating the current model.
			referenceOutput = correctOutput.createNewObject();
			inferenceImpl.partialInference(model, input, correctOutput,
					referenceOutput);
		}

		ExampleOutput lossAugmentedPredictedOutput = input.createOutput();
		if (lossNonAnnotatedWeight < 0)
			// Infer the whole output structure using the loss function. This is
			// the "better" output structure used to update the model.
			inferenceImpl.lossAugmentedInference(model, input, referenceOutput,
					lossAugmentedPredictedOutput, -lossAnnotatedWeight);
		else
			// Predict the best output structure to the current input structure
			// using a loss-augmented objective function that uses different
			// weights for annotated and non-annotated tokens.
			inferenceImpl.lossAugmentedInference(model, input, correctOutput,
					referenceOutput, lossAugmentedPredictedOutput,
					-lossAnnotatedWeight, -lossNonAnnotatedWeight);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, lossAugmentedPredictedOutput,
				predictedOutput, getCurrentLearningRate());

		// TODO debug
		if (DebugUtil.print && loss != 0d)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) lossAugmentedPredictedOutput,
					(SequenceOutput) predictedOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged
		// weights.
		model.sumUpdates(iteration);

		return loss;

	}

}
