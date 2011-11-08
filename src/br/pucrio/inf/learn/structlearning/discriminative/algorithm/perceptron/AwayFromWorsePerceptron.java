package br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron;

import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * McAllester et al.'s Perceptron implementation that uses a modified updating
 * rule which is proved to directly optimize the used loss function.
 * 
 * The difference from the away-from-worse implementation (this one) to the
 * toward-better implementation is that the former uses a loss-augmented
 * inference that privileges high-loss solutions and updates the model weights
 * away from the loss-augmented solution that is worse than the
 * non-loss-augmented solution.
 * 
 * On the other hand, the toward-better implementation privileges low-loss
 * solutions and updates the model weights toward this better solution.
 * 
 * @author eraldof
 * 
 */
public class AwayFromWorsePerceptron extends LossAugmentedPerceptron {

	public AwayFromWorsePerceptron(Inference taskImpl, Model initialModel) {
		super(taskImpl, initialModel);
	}

	public AwayFromWorsePerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate, double lossWeight,
			boolean randomize, boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				lossWeight, randomize, averageWeights,
				learningRateUpdateStrategy);
	}

	public AwayFromWorsePerceptron(Inference taskImpl, Model initialModel,
			int numberOfIterations, double learningRate,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight,
			double lossNonAnnotatedWeightInc, boolean randomize,
			boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		super(taskImpl, initialModel, numberOfIterations, learningRate,
				lossAnnotatedWeight, lossNonAnnotatedWeight,
				lossNonAnnotatedWeightInc, randomize, averageWeights,
				learningRateUpdateStrategy);
	}

	@Override
	public double train(ExampleInput input, ExampleOutput correctOutput,
			ExampleOutput predictedOutput) {

		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			// If the user asked to consider partially-labeled examples then
			// infer the missing values within the given correct output
			// structure before updating the current model.
			referenceOutput = correctOutput.createNewObject();
			inferenceImpl.partialInference(model, input, correctOutput,
					referenceOutput);
		}

		ExampleOutput lossAugmentedPredictedOutput = correctOutput
				.createNewObject();
		if (lossNonAnnotatedWeight < 0)
			// Infer the whole output structure using the loss function. This is
			// the "worse" output structure used to update the model.
			inferenceImpl.lossAugmentedInference(model, input, referenceOutput,
					lossAugmentedPredictedOutput, lossWeight);
		else
			// Predict the best output structure to the current input structure
			// using a loss-augmented objective function that uses different
			// weights for annotated and non-annotated tokens.
			inferenceImpl.lossAugmentedInferenceWithPartiallyLabeledReference(model, input, correctOutput,
					referenceOutput, lossAugmentedPredictedOutput,
					lossWeight, lossNonAnnotatedWeight);

		// Infer the whole output structure. This is the "better" output
		// structure used to update the model.
		inferenceImpl.inference(model, input, predictedOutput);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, predictedOutput,
				lossAugmentedPredictedOutput, getCurrentLearningRate());

		// TODO debug
		if (DebugUtil.print && loss != 0d)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) predictedOutput,
					(SequenceOutput) lossAugmentedPredictedOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged
		// weights.
		model.sumUpdates(iteration);
		
		++iteration;

		return loss;

	}

}
