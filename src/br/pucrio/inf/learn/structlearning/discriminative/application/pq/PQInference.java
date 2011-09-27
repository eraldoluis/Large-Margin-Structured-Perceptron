package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

public class PQInference implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		// TODO Auto-generated method stub

	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		// TODO Auto-generated method stub

	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		// TODO Auto-generated method stub

	}

	@Override
	public void lossAugmentedInferenceWithPartiallyLabeledReference(
			Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		// TODO Auto-generated method stub

	}

}
