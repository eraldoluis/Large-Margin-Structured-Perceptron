package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.maxbranching.MaximumBranchingAlgorithm;

/**
 * Inference algorithm for coreference resolution that is based on latent
 * rooted-tree structure. Coreference resolution consits in clustering a given
 * set of mentions. In this approach, each cluster of mentions additionally
 * comprises a latent rooted tree over its mentions.
 * 
 * @author eraldo
 * 
 */
public class CoreferenceMaxBranchInference implements Inference {

	/**
	 * Index of an artificial mention that is not within any cluster. Thus, to
	 * include an edge from this mention to any other (real) mention does not
	 * implicate cluster union. If this value is less than zero, then it is
	 * ignored.
	 */
	private int root;

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
	 * @param root
	 *            index of an artificial mention that is not within any cluster.
	 *            Thus, to include an edge from this mention to any other (real)
	 *            mention does not implicate cluster union. If this value is
	 *            less than zero, then it is ignored.
	 */
	public CoreferenceMaxBranchInference(int maxNumberOfTokens, int root) {
		this.root = root;
		maxBranchingAlgorithm = new MaximumBranchingAlgorithm(maxNumberOfTokens);
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((DPModel) model, (DPInput) input, (CorefOutput) output);
	}

	/**
	 * Inference method with objects of converted types.
	 * 
	 * @param model
	 * @param input
	 * @param output
	 */
	private void inference(DPModel model, DPInput input, CorefOutput output) {
		// Fill the graph weights.
		fillGraph(model, input);

		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		maxBranchingAlgorithm.findMaxBranching(input.getNumberOfTokens(),
				graph, output.getInvertedBranchingArray());

		// Compute the clustering from the found rooted tree.
		output.computeClusteringFromTree(root);
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
		for (int leftMention = 0; leftMention < numTokens; ++leftMention) {
			// Backward edges are not used.
			for (int rightMention = 0; rightMention <= leftMention; ++rightMention)
				graph[leftMention][rightMention] = Double.NaN;
			// Forward edges.
			for (int rightMention = leftMention + 1; rightMention < numTokens; ++rightMention)
				graph[leftMention][rightMention] = model.getEdgeScore(input,
						leftMention, rightMention);
		}
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		partialInference((DPModel) model, (DPInput) input,
				(CorefOutput) partiallyLabeledOutput,
				(CorefOutput) predictedOutput);
	}

	public void partialInference(DPModel model, DPInput input,
			CorefOutput partiallyLabeledOutput, CorefOutput predictedOutput) {
		// Fill edge weights (not including incorrect edges).
		fillPartialGraph(model, input, partiallyLabeledOutput);

		maxBranchingAlgorithm.setCheckUniqueRoot(false);

		int numTokens = input.getNumberOfTokens();

		/*
		 * Find the maximum branching and fill the output inverted branching
		 * array with it.
		 */
		maxBranchingAlgorithm.findMaxBranching(numTokens, graph,
				predictedOutput.getInvertedBranchingArray());

		maxBranchingAlgorithm.setCheckUniqueRoot(true);

		// Connect root nodes of the branching to the artificial root node.
		for (int tkn = 0; tkn < numTokens; ++tkn) {
			if (tkn == root)
				continue;
			if (predictedOutput.getHead(tkn) < 0)
				predictedOutput.setHead(tkn, root);
		}

		// Set the correct clustering for the predicted output structure.
		predictedOutput.setClusteringEqualTo(partiallyLabeledOutput);
	}

	/**
	 * Fill the graph using the given model and input sentence so that no
	 * incorrect edge is included in the graph. The reference output indicates
	 * which are the correct cluters and, thus, indicates which are the correct
	 * edges (intracluster edges).
	 * 
	 * @param model
	 * @param input
	 * @param referenceOutput
	 * @return
	 */
	private void fillPartialGraph(DPModel model, DPInput input,
			CorefOutput referenceOutput) {
		// Number of tokens in the input structure.
		int numTokens = input.getNumberOfTokens();
		// Fill the weight matrix.
		for (int mentionLeft = 0; mentionLeft < numTokens; ++mentionLeft) {
			// Backward edges are not used.
			for (int mentionRight = 0; mentionRight <= mentionLeft; ++mentionRight)
				graph[mentionLeft][mentionRight] = Double.NaN;
			// Forward edges.
			for (int mentionRight = mentionLeft + 1; mentionRight < numTokens; ++mentionRight) {
				// Only include correct edges (intracluster edges).
				if (referenceOutput.getClusterId(mentionLeft) == referenceOutput
						.getClusterId(mentionRight))
					graph[mentionLeft][mentionRight] = model.getEdgeScore(
							input, mentionLeft, mentionRight);
				else
					graph[mentionLeft][mentionRight] = Double.NaN;
			}
		}
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((DPModel) model, (DPInput) input,
				(CorefOutput) referenceOutput, (CorefOutput) predictedOutput,
				lossWeight);
	}

	public void lossAugmentedInference(DPModel model, DPInput input,
			CorefOutput referenceOutput, CorefOutput predictedOutput,
			double lossWeight) {
		// Number of tokens in the input structure.
		int numTokens = input.getNumberOfTokens();

		// Fill the graph weight matrix.
		fillGraph(model, input);

		// Add loss values.
		if (lossWeight != 0d) {
			for (int leftMention = 0; leftMention < numTokens; ++leftMention) {
				// Skip the root node edges.
				if (leftMention == root)
					// TODO use reference root edges in margin.
					continue;

				for (int rightMention = leftMention + 1; rightMention < numTokens; ++rightMention) {
					int correctLeftCluster = referenceOutput
							.getClusterId(leftMention);
					int correctRightCluster = referenceOutput
							.getClusterId(rightMention);

					// Skip mentions in the same cluster (intracluster edges).
					if (correctLeftCluster == correctRightCluster)
						continue;

					// Increment weight of incorrect edges (intercluster edges).
					graph[leftMention][rightMention] += lossWeight;
				}
			}
		}

		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		maxBranchingAlgorithm.findMaxBranching(input.getNumberOfTokens(),
				graph, predictedOutput.getInvertedBranchingArray());

		// Set the correct clustering for the predicted output structure.
		predictedOutput.setClusteringEqualTo(referenceOutput);
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

}
