package br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

public class PassiveAgressive2 extends PassiveAgressive {

	private double c;

	public PassiveAgressive2(Inference inferenceImpl, Model initialModel,
			int numberOfEpochs, boolean randomize, boolean averageWeights,
			double c) {
		super(inferenceImpl, initialModel, numberOfEpochs, randomize,
				averageWeights);

		this.c = c;
	}

	private double calculateTau(final double sufferLoss, final ExampleInput input,
			final ExampleOutput correctOutput, final ExampleOutput predictedOutput) {
		return sufferLoss/(correctOutput.getFeatureVectorLengthSquared(input, predictedOutput) +
				1/(2*c));
	}
}
