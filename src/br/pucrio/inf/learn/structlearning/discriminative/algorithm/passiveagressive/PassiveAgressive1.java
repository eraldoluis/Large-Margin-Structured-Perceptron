package br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

public class PassiveAgressive1 extends PassiveAgressive {
	
	private double c;

	public PassiveAgressive1(Inference inferenceImpl, Model initialModel,
			int numberOfEpochs, boolean randomize, boolean averageWeights
			, double c) {
		super(inferenceImpl, initialModel, numberOfEpochs, randomize, averageWeights);
		
		this.c = c;
	}
	
	protected double calculateTau(final double sufferLoss, final ExampleInput input,
			final ExampleOutput correctOutput, final ExampleOutput predictedOutput) {
		return Math.min(this.c, super.calculateTau(sufferLoss, input, correctOutput, predictedOutput));
	}

}
