package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Arrays;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.rank.RankOutput.WeightedItem;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Prediction algorithm for ranking functions. The ranking model gives one
 * weight for each item within a query. Thus, the prediction consists only in
 * sorting the items according to the given weights.
 * 
 * @author eraldo
 * 
 */
public class Bisectionnference implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((BisectionModel) model, (BisectionInput) input, (BisectionOutput) output);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((BisectionModel) model, (BisectionInput) input,
				(BisectionOutput) referenceOutput, (BisectionOutput) predictedOutput,
				lossWeight);
	}

	/**
	 * The inference for ranking models is very simple. Each item is given a
	 * weight according to its features weights (given by the model). Then, we
	 * just sort the items based on their weights.
	 * 
	 * @param model
	 * @param input
	 * @param output
	 */
	public void inference(BisectionModel model, BisectionInput input, BisectionOutput output) {
		fillWeights(model, input, output, null, 0d);
		Arrays.sort(output.weightedPapers);
	}

	/**
	 * Fill the weight array in the given output according to the corresponding
	 * input features and their weights in the given model.
	 * 
	 * @param model
	 * @param input
	 * @param output
	 */
	private void fillWeights(BisectionModel model, BisectionInput input,
			BisectionOutput output, BisectionOutput correct, double lossWeight) {
		// Number of items for this query.
		int size = input.size();
		for (int idxItem = 0; idxItem < size; ++idxItem) {
			// Get the weighted item at the current index.
			WeightedItem wItem = output.weightedPapers[idxItem];
			// Initialize the current item weight.
			wItem.weight = 0;
			// Compute sum of item features weights.
			int[] ftrs = input.getFeatures(wItem.item);
			for (int ftr : ftrs)
				wItem.weight += model.getFeatureWeight(ftr);
		}

		if (correct != null && lossWeight != 0d) {
			for (int idxItem = 0; idxItem < size; ++idxItem) {
				// Get the weighted item at the current index.
				WeightedItem wItem = output.weightedPapers[idxItem];
				if (!correct.isConfirmed(wItem.item))
					// Increase weight.
					wItem.weight += lossWeight;
			}
		}
	}

	public void lossAugmentedInference(BisectionModel model, BisectionInput input,
			BisectionOutput referenceOutput, BisectionOutput predictedOutput,
			double lossWeight) {
		fillWeights(model, input, predictedOutput, referenceOutput, lossWeight);
		Arrays.sort(predictedOutput.weightedPapers);
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		throw new NotImplementedException();
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

}
