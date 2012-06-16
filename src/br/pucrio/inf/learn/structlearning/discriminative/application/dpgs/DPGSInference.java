package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Dependency parser with grandparent and siblings features based on dual
 * decomposition as in Koo et al. (EMNLP-2010).
 * 
 * This algorithm uses two sub-solvers: maximum branching to find a rooted tree
 * and a dynamic program that considers the sencod-order features (grandparent
 * and siblings). The dual decomposition, through Lagrangean relaxation,
 * guarantees that the solution of the later solver respects rooted tree
 * constraints with no need to directly consider these constraints in the
 * dynamic program, which would be very difficult, since that is a NP-hard
 * problem.
 * 
 * @author eraldo
 * 
 */
public class DPGSInference implements Inference {

	public void inference(DPGSModel model, DPGSInput input, DPGSOutput output) {
	}

	public void partialInference(DPGSModel model, DPGSInput input,
			DPGSOutput partiallyLabeledOutput, DPGSOutput predictedOutput) {
	}

	public void lossAugmentedInference(DPGSModel model, DPGSInput input,
			DPGSOutput referenceOutput, DPGSOutput predictedOutput,
			double lossWeight) {
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((DPGSModel) model, (DPGSInput) input, (DPGSOutput) output);
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		partialInference((DPGSModel) model, (DPGSInput) input,
				(DPGSOutput) partiallyLabeledOutput,
				(DPGSOutput) predictedOutput);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((DPGSModel) model, (DPGSInput) input,
				(DPGSOutput) referenceOutput, (DPGSOutput) predictedOutput,
				lossWeight);
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

}
