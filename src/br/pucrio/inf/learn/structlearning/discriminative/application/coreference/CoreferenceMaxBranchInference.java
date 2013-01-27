package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.maxbranching.DirectedMaxBranchAlgorithm;
import br.pucrio.inf.learn.util.maxbranching.MaximumBranchingAlgorithm;
import br.pucrio.inf.learn.util.maxbranching.UndirectedMaxBranchAlgorithm;

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
	 * Strategy to be used to solve the inference problem.
	 * 
	 * @author eraldo
	 */
	public enum InferenceStrategy {
		/**
		 * Fixed maximum branching. The correct output structures in training
		 * data indicate the (only one) correct tree.
		 */
		BRANCH,

		/**
		 * Latent maximum branching. The correct output structures in training
		 * data indicate only the correct cluster. The underlying tree is
		 * assumed as latent and predicted by Chu-Liu-Edmonds algorithm.
		 */
		LBRANCH,

		/**
		 * Latent (undirected) MST. The correct output structures in training
		 * data indicate only the correct cluster. The underlying tree is
		 * assumed as latent and predicted by Kruskal algorithm.
		 */
		LKRUSKAL,
	}

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
	 * Multicative factor to be used only on edges that come from the artificial
	 * root node. This factor can be used to bias the updates in order to
	 * stimulate or distimulate the selection of these edges by the model.
	 */
	private double lossFactorForRootEdges;

	/**
	 * Whether to use the artificial root node or not.
	 */
	private boolean useRoot;

	public double treeWeight;

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
	public CoreferenceMaxBranchInference(int maxNumberOfTokens, int root,
			InferenceStrategy inferenceStrategy) {
		this.root = root;
		if (inferenceStrategy != InferenceStrategy.LKRUSKAL)
			maxBranchingAlgorithm = new DirectedMaxBranchAlgorithm(
					maxNumberOfTokens);
		else
			maxBranchingAlgorithm = new UndirectedMaxBranchAlgorithm(
					maxNumberOfTokens);
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
		lossFactorForRootEdges = -1d;
		useRoot = true;
		/*
		 * If the artificial root node is used, then the prediction algorithm
		 * can consider even negative-weight edges, since there is always the
		 * root node as an option. Otherwise, negative-weight edges must be
		 * avoided in order to allow the prediction of non-connected trees (one
		 * tree for each cluster).
		 */
		maxBranchingAlgorithm.setOnlyPositiveEdges(!useRoot);
	}

	/**
	 * Set whether to use the artificial root node or not.
	 * 
	 * @param useRoot
	 */
	public void setUseRoot(boolean useRoot) {
		this.useRoot = useRoot;
		/*
		 * If the artificial root node is used, then the prediction algorithm
		 * can consider even negative-weight edges, since there is always the
		 * root node as an option. Otherwise, negative-weight edges must be
		 * avoided in order to allow the prediction of non-connected trees (one
		 * tree for each cluster).
		 */
		maxBranchingAlgorithm.setOnlyPositiveEdges(!useRoot);
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
		treeWeight = maxBranchingAlgorithm.findMaxBranching(
				input.getNumberOfTokens(), graph,
				output.getInvertedBranchingArray());

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
		if (graph.length < numTokens) {
			graph = new double[numTokens][numTokens];
			maxBranchingAlgorithm.realloc(numTokens);
		}
		// Fill the weight matrix.
		for (int leftMention = 0; leftMention < numTokens; ++leftMention) {
			for (int rightMention = 0; rightMention < numTokens; ++rightMention) {
				if (!useRoot && (leftMention == root || rightMention == root))
					graph[leftMention][rightMention] = Double.NaN;
				else
					graph[leftMention][rightMention] = model.getEdgeScore(
							input, leftMention, rightMention);
			}
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

		// Save current flag for unique root check.
		boolean oldCheckFlag = maxBranchingAlgorithm.isCheckUniqueRoot();
		// Turn off unique root check.
		maxBranchingAlgorithm.setCheckUniqueRoot(false);

		// Save current flag to positive edges only.
		boolean oldOnlyPosEdges = maxBranchingAlgorithm.isOnlyPositiveEdges();
		/*
		 * Turn off only positive edges to avoid disconnected coreferring
		 * clusters due to lack of positive edges.
		 */
		maxBranchingAlgorithm.setOnlyPositiveEdges(false);
		int numMentions = input.getNumberOfTokens();

		/*
		 * Find the maximum branching and fill the output inverted branching
		 * array with it.
		 */
		treeWeight = maxBranchingAlgorithm.findMaxBranching(numMentions, graph,
				predictedOutput.getInvertedBranchingArray());

		// Recover old value for unique root check flag.
		maxBranchingAlgorithm.setCheckUniqueRoot(oldCheckFlag);
		// Recover old value for positive edges only flag.
		maxBranchingAlgorithm.setOnlyPositiveEdges(oldOnlyPosEdges);

		if (useRoot) {
			// Connect root nodes of the branching to the artificial root node.
			for (int mention = 0; mention < numMentions; ++mention) {
				if (mention == root)
					continue;
				if (predictedOutput.getHead(mention) < 0)
					predictedOutput.setHead(mention, root);
			}
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
		if (graph.length < numTokens) {
			graph = new double[numTokens][numTokens];
			maxBranchingAlgorithm.realloc(numTokens);
		}
		// Fill the weight matrix.
		for (int mentionLeft = 0; mentionLeft < numTokens; ++mentionLeft) {
			// Left mention cluster id.
			int idClusterLeftMention = referenceOutput
					.getClusterId(mentionLeft);
			// Right mentions.
			for (int mentionRight = 0; mentionRight < numTokens; ++mentionRight) {
				// Right mention cluster id.
				int idClusterRightMention = referenceOutput
						.getClusterId(mentionRight);
				// Only include correct edges (intracluster edges).
				if (idClusterLeftMention == idClusterRightMention)
					graph[mentionLeft][mentionRight] = model.getEdgeScore(
							input, mentionLeft, mentionRight);
				else
					// Double.NaN indicates missing edge.
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
			/*
			 * Special loss weight for edges coming from the artificial root
			 * node.
			 */
			double rootLossWeight = lossWeight;
			if (lossFactorForRootEdges >= 0d)
				rootLossWeight *= lossFactorForRootEdges;

			for (int leftMention = 0; leftMention < numTokens; ++leftMention) {
				if (leftMention == root) {
					/*
					 * Root edges (edges linking right mentions to artificial
					 * root node.
					 */
					for (int rightMention = 0; rightMention < numTokens; ++rightMention)
						if (referenceOutput.getHead(rightMention) != root)
							// Increment weight for incorrect root edge.
							if (!Double.isNaN(graph[leftMention][rightMention]))
								graph[leftMention][rightMention] += rootLossWeight;
				} else {
					// Ordinary edges, i.e., edges between real mentions.
					for (int rightMention = 0; rightMention < numTokens; ++rightMention) {
						int correctLeftCluster = referenceOutput
								.getClusterId(leftMention);
						int correctRightCluster = referenceOutput
								.getClusterId(rightMention);
						if (correctLeftCluster != correctRightCluster)
							/*
							 * Increment weight of incorrect edges (intercluster
							 * edges).
							 */
							if (!Double.isNaN(graph[leftMention][rightMention]))
								graph[leftMention][rightMention] += lossWeight;
					}
				}
			}
		}

		/*
		 * Find the maximum branching rooted at the zero node and fill the
		 * output inverted branching array with it.
		 */
		treeWeight = maxBranchingAlgorithm.findMaxBranching(
				input.getNumberOfTokens(), graph,
				predictedOutput.getInvertedBranchingArray());

		// Set the correct clustering for the predicted output structure.
		// predictedOutput.setClusteringEqualTo(referenceOutput);
		predictedOutput.computeClusteringFromTree(root);
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

	/**
	 * Set the loss factor for edges that come from the artificial root node.
	 * 
	 * @param factor
	 */
	public void setLossFactorForRootEdges(double factor) {
		this.lossFactorForRootEdges = factor;
	}

	public static void printIncorrectClusters(CorefModel model,
			CorefInput input, CorefOutput correct, CorefOutput predicted) {
		Map<Integer, ? extends Set<Integer>> explicitClusteringCorrect = createExplicitClustering(correct);
		Map<Integer, ? extends Set<Integer>> explicitClusteringPredicted = createExplicitClustering(predicted);

		System.out.print("Correct clustering: ");
		for (Set<Integer> cluster : explicitClusteringCorrect.values()) {
			// Skip completely correct clusters.
			int oneMentionCorrect = cluster.iterator().next();
			int idCorrect = correct.getClusterId(oneMentionCorrect);
			int idPredicted = predicted.getClusterId(idCorrect);
			Set<Integer> clusterPredicted = explicitClusteringPredicted
					.get(idPredicted);
			if (cluster.equals(clusterPredicted))
				continue;

			System.out.print("{");
			for (int m : cluster)
				System.out.print(m + ",");
			System.out.print("} ");
		}
		System.out.println("\n");

		System.out.print("Predicted clustering: ");
		for (Set<Integer> cluster : explicitClusteringPredicted.values()) {
			// Skip completely correct clusters.
			int oneMentionPredicted = cluster.iterator().next();
			int idPredicted = predicted.getClusterId(oneMentionPredicted);
			int idCorrect = correct.getClusterId(idPredicted);
			Set<Integer> clusterCorrect = explicitClusteringCorrect
					.get(idCorrect);
			if (cluster.equals(clusterCorrect))
				continue;

			System.out.print("{");
			for (int m : cluster)
				System.out.print(m + " ");
			System.out.print("} ");
		}
		System.out.println("\n");
	}

	public void printEdgesOfIncorrectMentions(CorefModel model,
			CorefInput input, CorefOutput correct, CorefOutput predicted) {
		// Explicit clusters.
		Map<Integer, ? extends Set<Integer>> explicitClusteringCorrect = createExplicitClustering(correct);
		Map<Integer, ? extends Set<Integer>> explicitClusteringPredicted = createExplicitClustering(predicted);

		// Fill graph weights.
		fillGraph(model, input);

		int numMentions = correct.size();
		for (int idxMention = 0; idxMention < numMentions; ++idxMention) {
			int clusterIdCorrect = correct.getClusterId(idxMention);
			int clusterIdPredicted = predicted.getClusterId(idxMention);

			Set<Integer> clusterCorrect = explicitClusteringCorrect
					.get(clusterIdCorrect);
			Set<Integer> clusterPredicted = explicitClusteringPredicted
					.get(clusterIdPredicted);

			if (!clusterCorrect.equals(clusterPredicted)) {
				int leftMentionCorrect = correct.getHead(idxMention);
				int leftMentionPredicted = predicted.getHead(idxMention);
				System.out.println(String.format(
						"Correct: %d>%d (%f) | Predicted: %d>%d (%f)",
						leftMentionCorrect, idxMention,
						graph[leftMentionCorrect][idxMention],
						leftMentionPredicted, idxMention,
						graph[leftMentionPredicted][idxMention]));
			}
		}
		System.out.println("\n");
	}

	/**
	 * For each cluster in <code>output</code>, create a set with its mentions.
	 * Mentions are represented by their integer ids. The set of clusters are
	 * represented by a map whose keys are the clusters ids, i.e., the id of one
	 * specific mention in the cluster (the representant mention).
	 * 
	 * @param output
	 * @return
	 */
	public static Map<Integer, ? extends Set<Integer>> createExplicitClustering(
			CorefOutput output) {
		HashMap<Integer, TreeSet<Integer>> explicitClustering = new HashMap<Integer, TreeSet<Integer>>();
		int numMentions = output.size();
		for (int idxMention = 0; idxMention < numMentions; ++idxMention) {
			// Get the current mention id.
			int id = output.getClusterId(idxMention);
			// Get the set of mentions in the current mention cluster.
			TreeSet<Integer> cluster = explicitClustering.get(id);
			if (cluster == null) {
				// First mention of its cluster.
				cluster = new TreeSet<Integer>();
				explicitClustering.put(id, cluster);
			}
			// Add the current mention to its cluster set.
			cluster.add(idxMention);
		}
		return explicitClustering;
	}

}
