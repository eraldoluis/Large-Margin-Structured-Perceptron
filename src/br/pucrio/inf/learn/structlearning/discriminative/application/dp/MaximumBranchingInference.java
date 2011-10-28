package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.maxbranching.CompleteGraph;
import br.pucrio.inf.learn.util.maxbranching.MaximumBranchingAlgorithm;

/**
 * Inference algorithm for dependency parsing problems. It corresponds to
 * finding a maximum branching on the complete graph whose nodes are tokens of a
 * sentence and the edge weights are given by the sum of the features weights in
 * each edge (given by the model).
 * 
 * @author eraldo
 * 
 */
public class MaximumBranchingInference implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((DPModel) model, (DPInput) input, (DPOutput) output);
	}

	private void inference(DPModel model, DPInput input, DPOutput output) {
		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		new MaximumBranchingAlgorithm().getMaxBranching(
				createGraph(model, input), 0,
				output.getInvertedBranchingArray());
	}

	/**
	 * Create a complete graph using the given model and input sentence.
	 * 
	 * @param model
	 * @param input
	 * @return
	 */
	private CompleteGraph createGraph(DPModel model, DPInput input) {
		// Number of tokens in the input structure.
		int numTokens = input.getNumberOfTokens();

		// Allocate and fill the weight matrix.
		double[][] weights = new double[numTokens][numTokens];
		for (int head = 0; head < numTokens; ++head)
			// Root node (zero) is never considered a dependent.
			for (int dependent = 1; dependent < numTokens; ++dependent)
				for (int ftrCode : input.getFeatureCodes(head, dependent))
					weights[head][dependent] += model.getFeatureWeight(ftrCode);

		// Create a complete graph using these weights.
		return new CompleteGraph(weights);
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		throw new NotImplementedException();
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((DPModel) model, (DPInput) input,
				(DPOutput) referenceOutput, (DPOutput) predictedOutput,
				lossWeight);
	}

	public void lossAugmentedInference(DPModel model, DPInput input,
			DPOutput referenceOutput, DPOutput predictedOutput,
			double lossWeight) {
		// Number of tokens in the input structure.
		int numTokens = input.getNumberOfTokens();

		// Create the lossless graph.
		CompleteGraph graph = createGraph(model, input);

		// Add loss values.
		for (int head = 0; head < numTokens; ++head) {
			for (int dependent = 1; dependent < numTokens; ++dependent) {
				// Skip self-loops.
				if (head == dependent)
					continue;
				// Skip correct labeled tokens.
				if (head == referenceOutput.getHead(dependent))
					continue;
				// Increment edge weight for misclassifing edges.
				graph.incEdgeWeight(head, dependent, lossWeight);
			}
		}

		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		new MaximumBranchingAlgorithm().getMaxBranching(graph, 0,
				predictedOutput.getInvertedBranchingArray());
	}

	@Override
	public void lossAugmentedInferenceWithPartiallyLabeledReference(
			Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

}
