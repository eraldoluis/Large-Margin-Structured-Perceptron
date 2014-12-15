package br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive;

import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.Perceptron;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.DebugUtil;

public class PassiveAgressive extends Perceptron {

	/**
	 * If <code>true</code> then consider partially-annotated examples.
	 */
	protected boolean partiallyAnnotatedExamples;
	private PassiveAgressiveUpdate passiveAgressiveUpdate;

	/**
	 * Create a PassiveAgressive to train the given initial model using the
	 * given number of iterations.
	 * 
	 * @param inferenceImpl
	 * @param initialModel
	 * @param numberOfEpochs
	 * @param learningRate
	 * @param randomize
	 * @param averageWeights
	 * @param learningRateUpdateStrategy
	 */
	public PassiveAgressive(Inference inferenceImpl, Model initialModel,
			int numberOfEpochs, boolean randomize, boolean averageWeights,
			PassiveAgressiveUpdate passiveAgressiveUpdate) {

		super(inferenceImpl, initialModel, numberOfEpochs, 1.0, randomize,
				averageWeights, LearnRateUpdateStrategy.NONE);

		this.passiveAgressiveUpdate = passiveAgressiveUpdate;
	}

	/**
	 * Create a PassiveAgressive to train the given initial model using the
	 * given number of iterations.
	 * 
	 * @param inferenceImpl
	 * @param initialModel
	 * @param numberOfEpochs
	 * @param learningRate
	 * @param randomize
	 * @param averageWeights
	 * @param learningRateUpdateStrategy
	 */
	public PassiveAgressive(Inference inferenceImpl, Model initialModel,
			int numberOfEpochs, boolean randomize, boolean averageWeights) {
		super(inferenceImpl, initialModel, numberOfEpochs, 1.0, randomize,
				averageWeights, LearnRateUpdateStrategy.NONE);
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

		// Predict the best output with the current mobel.
		inferenceImpl.inference(model, input, predictedOutput);

		double sufferLoss = inferenceImpl.calculateSufferLoss(referenceOutput, predictedOutput,
				passiveAgressiveUpdate);
		
		double tau = calculateTau(sufferLoss, input, correctOutput, predictedOutput);
		
		// Update the current model and return the loss for this example.
		double loss = model.update(input, referenceOutput, predictedOutput,
				tau);

		// Debug.
		if (DebugUtil.print)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) referenceOutput,
					(SequenceOutput) predictedOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged weights.
		model.sumUpdates(iteration);

		++iteration;

		return loss;
	}

	protected double calculateTau(double sufferLoss, ExampleInput input,
			ExampleOutput correctOutput, ExampleOutput predictedOutput) {
		double len = correctOutput.getFeatureVectorLengthSquared(input, predictedOutput);
		
		if(len == 0.0d)
			return 0.0d;
		
		return sufferLoss / len;
	}
}
