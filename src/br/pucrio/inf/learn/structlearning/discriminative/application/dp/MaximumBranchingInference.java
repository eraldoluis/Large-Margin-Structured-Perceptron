package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
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

	/**
	 * Algorithm and its data structures for finding maximum branching.
	 */
	private MaximumBranchingAlgorithm maxBranchingAlgorithm;

	/**
	 * Input graph used to predict the maximum branching.
	 */
	private double[][] graph;

	/**
	 * Create an inference implementation to deal with sentences that have the
	 * given maximum number of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public MaximumBranchingInference(int maxNumberOfTokens) {
		maxBranchingAlgorithm = new MaximumBranchingAlgorithm(maxNumberOfTokens);
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((DPModel) model, (DPInput) input, (DPOutput) output);
	}

	private void inference(DPModel model, DPInput input, DPOutput output) {
		// Fill the graph weights.
		fillGraph(model, input);

		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		maxBranchingAlgorithm.findMaxBranching(input.getNumberOfTokens(),
				graph, output.getInvertedBranchingArray());
	}

	/**
	 * Fill the graph using the given model and input sentence.
	 * 
	 * @param model
	 * @param input
	 * @return
	 */
	private void fillGraph(DPModel model, DPInput input) {
		// Number of tokens in the input structure.
		int numTokens = input.getNumberOfTokens();
		// Fill the weight matrix.
		for (int head = 0; head < numTokens; ++head)
			for (int dependent = 0; dependent < numTokens; ++dependent)
				graph[head][dependent] = model.getEdgeScore(input, head,
						dependent);
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

		// Fill the graph weight matrix.
		fillGraph(model, input);

		// Add loss values.
		for (int dependent = 1; dependent < numTokens; ++dependent) {
			int correctHead = referenceOutput.getHead(dependent);
			if (correctHead == -1)
				/*
				 * Skip tokens with no correct edge (due to pruning
				 * preprocessing).
				 */
				continue;

			for (int head = 0; head < numTokens; ++head) {
				// Skip self-loops.
				if (head == dependent)
					continue;
				// Skip correct labeled tokens.
				if (head == correctHead)
					continue;
				graph[head][dependent] += lossWeight;
			}
		}

		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		maxBranchingAlgorithm.findMaxBranching(input.getNumberOfTokens(),
				graph, predictedOutput.getInvertedBranchingArray());
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
