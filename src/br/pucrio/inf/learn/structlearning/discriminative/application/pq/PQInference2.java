package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQModel;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

public class PQInference2 implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		// TODO Auto-generated method stub
		inference((PQModel) model, (PQInput) input, (PQOutput) output);
	}
	
	public void inference(PQModel model, PQInput input, PQOutput output) {
		int numberOfPersons = input.size();
		double maxInnerProduct = -10000000d;
		int personChosen = -1;
		double noise = 0.000001;
		
		for (int i = 0; i < numberOfPersons; i++) {
			int[] personFeatures = input.getPersonFeatures(i);
			
			double innerProduct = 0d;
			for (int j = 0; j < personFeatures.length; j++) {
				double w = model.getFeatureWeight(personFeatures[j]);
				if (w > noise)
					innerProduct += w;
			}
			
			if (innerProduct > maxInnerProduct) {
				maxInnerProduct = innerProduct;
				personChosen   = i;
			}
		}
		
		output.setPerson(personChosen);
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
