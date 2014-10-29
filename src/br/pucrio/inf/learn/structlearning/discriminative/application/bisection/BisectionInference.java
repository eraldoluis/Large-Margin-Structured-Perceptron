package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive.PassiveAgressiveUpdate;
import br.pucrio.inf.learn.structlearning.discriminative.application.bisection.BisectionOutput.WeightedPaper;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.maxbranching.KruskalAlgorithm;
import br.pucrio.inf.learn.util.maxbranching.SimpleWeightedEdge;

/**
 * Prediction algorithm for bisection model. Based on Kruskal algorithm to
 * cluster candidate papers in confirmed and deleted sets. Then, use distance to
 * the artificial deleted node (paper at the index zero) to rank papers within
 * each cluster.
 * 
 * @author eraldo
 * 
 */
public class BisectionInference implements Inference {

	/**
	 * Implementation of Kruskal algorithm to find MST.
	 */
	private KruskalAlgorithm mstAlgorithm;

	/**
	 * Factor to multiply the loss value for arcs from the artificial DELETED
	 * paper.
	 */
	private double deletedPaperLossFactor;

	/**
	 * Artificial edges used in the prediction of the latent structure.
	 */
	private ArrayList<SimpleWeightedEdge> artificialEdges;

	/**
	 * Comparator of WeightedPaper's that ranks confirmed papers before deleted
	 * ones and within each class ranks according to the inverse order of paper
	 * weights.
	 */
	private static final Comparator<WeightedPaper> compPapers = new Comparator<BisectionOutput.WeightedPaper>() {
		@Override
		public int compare(WeightedPaper p1, WeightedPaper p2) {
			if (p1.confirmed != p2.confirmed) {
				if (p1.confirmed)
					return -1;
				return 1;
			}
			if (p1.weight > p2.weight)
				return -1;
			if (p1.weight < p2.weight)
				return 1;
			return 0;
		}
	};

	public BisectionInference() {
		deletedPaperLossFactor = 1d;
		mstAlgorithm = new KruskalAlgorithm(0, 2);
		artificialEdges = new ArrayList<SimpleWeightedEdge>();
	}

	public void setOnlyPositiveEdges(boolean val) {
		mstAlgorithm.setOnlyPositiveEdges(val);
	}

	public void setDeletedPaperLossFactor(double factor) {
		deletedPaperLossFactor = factor;
	}

	/**
	 * Predict a ranking of the given instance papers.
	 * 
	 * First, predict the latent MST structure over the input papers. Second,
	 * split papers into confirmed or deleted according to the predicted MST
	 * (papers connected to the artificial deleted node are considered deleted,
	 * all the remaining papers are considered confirmed). Finally, sort papers
	 * within each split by the inverse distance to the artificial deleted
	 * paper.
	 * 
	 * @param model
	 * @param input
	 * @param output
	 */
	public void inference(BisectionModel model, BisectionInput input,
			BisectionOutput output) {
		// Structure size.
		int size = input.size();
		// Build graph to run MST algorithm.
		double[][] graph = new double[size][size];
		fillGraph(graph, model, input, null, 0d);

		// Guarantee that auxiliary data structures fit this instance.
		mstAlgorithm.realloc(size);
		// Find MST.
		mstAlgorithm.findMaxBranching(size, graph, output.getMst(),
				output.getPartition());

		// Compute confirmed papers from the found MST.
		output.computeSplitFromMstPartition();

		// Sort edges within clusters.
		fillWeightsAndConfirmed(output, graph);
		Arrays.sort(output.weightedPapers, compPapers);
	}

	/**
	 * Predict the latent MST in the given predicted output. If a reference
	 * output is given, then add a margin (<code>lossWeight</code>) to each
	 * incorrect edge before running the MST algorithm.
	 * 
	 * The confirmed-deleted splits and the ranking are not computed.
	 * 
	 * @param model
	 * @param input
	 * @param referenceOutput
	 * @param predictedOutput
	 * @param lossWeight
	 */
	public void lossAugmentedInference(BisectionModel model,
			BisectionInput input, BisectionOutput referenceOutput,
			BisectionOutput predictedOutput, double lossWeight) {
		// Structure size.
		int size = input.size();
		// Build graph to run MST algorithm.
		double[][] graph = new double[size][size];
		fillGraph(graph, model, input, referenceOutput, lossWeight);

		// Guarantee that auxiliary data structures fit this instance.
		mstAlgorithm.realloc(size);
		// Find MST.
		mstAlgorithm.findMaxBranching(size, graph, predictedOutput.getMst(),
				predictedOutput.getPartition());

		// TODO I guess the following is not necessary during training.
		// // Compute confirmed papers from the found MST.
		// predictedOutput.computeSplitFromMst();
		//
		// // Sort edges within clusters.
		// fillWeightsAndConfirmed(predictedOutput, graph);
		// Arrays.sort(predictedOutput.weightedPapers);
	}

	/**
	 * Predict a latent structure (MST) for <code>predictedOutput</code> that
	 * respects the correct confirmed-deleted split within
	 * <code>partiallyLabeledOutput</code>. The predicted tree is used as the
	 * groun truth to update model parameters during training.
	 * 
	 * @param model
	 * @param input
	 * @param partiallyLabeledOutput
	 * @param predictedOutput
	 */
	private void partialInference(BisectionModel model, BisectionInput input,
			BisectionOutput partiallyLabeledOutput,
			BisectionOutput predictedOutput) {
		// Structure size.
		int size = input.size();
		/*
		 * Build graph to run MST algorithm. This graph is split in two
		 * disconnected components: confirmed and deleted. So that the found MST
		 * will always be split.
		 */
		double[][] graph = new double[size][size];
		fillPartiaGraph(graph, model, input, partiallyLabeledOutput);

		// Guarantee that auxiliary data structures fit this instance.
		mstAlgorithm.realloc(size);

		// Find MST.
		boolean val = mstAlgorithm.isOnlyPositiveEdges();
		mstAlgorithm.setOnlyPositiveEdges(false);
		mstAlgorithm.findMaxBranching(size, graph, predictedOutput.getMst(),
				predictedOutput.getPartition());
		mstAlgorithm.setOnlyPositiveEdges(val);

		// Copy confirmed papers structure.
		predictedOutput.setConfirmedPapersEqualTo(partiallyLabeledOutput);

		// TODO test
		// /*
		// * Connect the CONFIRMED artificial paper (1) to the first confirmed
		// * paper. And connect the DELETED artificial paper (0) to the first
		// * deleted paper.
		// */
		// artificialEdges.clear();
		// artificialEdges.ensureCapacity(2 * size);
		// for (int paper2 = 0; paper2 < size; ++paper2) {
		// // Add artificial edge from the DELETED paper.
		// if (!partiallyLabeledOutput.isConfirmed(paper2)) {
		// double w = getEdgeWeight(model, input, 0, paper2);
		// if (!Double.isNaN(w))
		// artificialEdges.add(new SimpleWeightedEdge(0, paper2, w));
		// } else {
		// // Add artificial edge from the CONFIRMED paper.
		// double w = getEdgeWeight(model, input, 1, paper2);
		// if (!Double.isNaN(w))
		// artificialEdges.add(new SimpleWeightedEdge(1, paper2, w));
		// }
		// }
		//
		// // Sort the artificial edges according to the weights given above.
		// Collections.sort(artificialEdges, KruskalAlgorithm.comp);
		//
		// // Add artificial edges to get exactly two connected components.
		// Set<SimpleWeightedEdge> mst = predictedOutput.getMst();
		// DisjointSets partition = predictedOutput.getPartition();
		// int clusterDel = partition.find(0);
		// int clusterConf = partition.find(1);
		// for (SimpleWeightedEdge e : artificialEdges) {
		// if (e.from == 0) {
		// int cluster = partition.find(e.to);
		// if (cluster != clusterDel) {
		// partition.union(clusterDel, cluster);
		// mst.add(e);
		// }
		// } else if (e.from == 1) {
		// int cluster = partition.find(e.to);
		// if (cluster != clusterConf) {
		// partition.union(clusterConf, cluster);
		// mst.add(e);
		// }
		// }
		// }
	}

	/**
	 * Fill the given graph with weights derived from the given input edge
	 * features, using the given model parameters. The graph is used to find the
	 * MST which will give rise to confirmed-deleted papers clustering.
	 * 
	 * @param graph
	 * @param model
	 * @param input
	 * @param correct
	 * @param lossWeight
	 */
	private void fillGraph(double[][] graph, BisectionModel model,
			BisectionInput input, BisectionOutput correct, double lossWeight) {
		int size = input.size();
		for (int paper1 = 0; paper1 < size; ++paper1) {
			for (int paper2 = 0; paper2 < size; ++paper2) {
				// Compute edge weight.
				double w = getEdgeWeight(model, input, paper1, paper2);

				if (Double.isNaN(w)) {
					// Skip inexistent edges.
					graph[paper1][paper2] = w;
					continue;
				}

				// Add margin for incorrect edges.
				if (correct != null
						&& lossWeight != 0d
						&& (correct.isConfirmed(paper1) != correct
								.isConfirmed(paper2))) {
					if (paper1 != 0 && paper2 != 0)
						w += lossWeight;
					else
						/*
						 * Add special margin for incorrect edge to the
						 * artificial DELETED paper.
						 */
						w += lossWeight * deletedPaperLossFactor;
				}

				graph[paper1][paper2] = w;
			}
		}
	}

	/**
	 * Fill a graph to run MST algorithm on it. Use the given correct output to
	 * avoid including incorrect edges. That is, include only correct edges.
	 * Thus, the predicted output is always correct.
	 * 
	 * @param graph
	 * @param model
	 * @param input
	 * @param correct
	 */
	private void fillPartiaGraph(double[][] graph, BisectionModel model,
			BisectionInput input, BisectionOutput correct) {
		int size = input.size();
		for (int paper1 = 0; paper1 < size; ++paper1) {
			for (int paper2 = 0; paper2 < size; ++paper2) {
				// Do not include incorrect edges.
				if (correct.isConfirmed(paper1) != correct.isConfirmed(paper2)) {
					graph[paper1][paper2] = Double.NaN;
					continue;
				}

				/*
				 * Do not include artificial edges in order to force the
				 * selection of ordinary edges.
				 */
				// TODO test
				// if (paper1 == 0 || paper1 == 1 || paper2 == 0 || paper2 == 1)
				// {
				// graph[paper1][paper2] = Double.NaN;
				// continue;
				// }

				// Compute edge weight.
				graph[paper1][paper2] = getEdgeWeight(model, input, paper1,
						paper2);
			}
		}
	}

	private double getEdgeWeight(BisectionModel model, BisectionInput input,
			int paper1, int paper2) {
		// Categorical features of this edge.
		int[] ftrCodes = input.getFeatureCodes(paper1, paper2);
		if (ftrCodes == null)
			return Double.NaN;

		// Numerical features of this edge.
		double[] ftrValues = input.getFeatureValues(paper1, paper2);

		// Compute edge weight.
		double w = 0d;
		for (int idxFtr = 0; idxFtr < ftrCodes.length; ++idxFtr)
			w += model.getFeatureWeight(ftrCodes[idxFtr]) * ftrValues[idxFtr];
		return w;
	}

	/**
	 * Fill the properties of the weighted papers in the given output. Each
	 * paper has a weight and a confirmed flag.
	 * 
	 * @param output
	 * @param graph
	 */
	private void fillWeightsAndConfirmed(BisectionOutput output,
			double[][] graph) {
		/*
		 * TODO test distance to clusters (confirmed and deleted) instead of
		 * distance to deleted node.
		 */
		int size = output.size();
		WeightedPaper[] wPapers = output.weightedPapers;
		for (int idx = 0; idx < size; ++idx) {
			WeightedPaper wPaper = wPapers[idx];
			int paper = wPaper.paper;
			if (output.isConfirmed(paper)) {
				wPaper.weight = graph[1][paper];
				wPaper.confirmed = true;
			} else {
				wPaper.weight = -graph[0][paper];
				wPaper.confirmed = false;
			}
		}
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((BisectionModel) model, (BisectionInput) input,
				(BisectionOutput) output);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((BisectionModel) model, (BisectionInput) input,
				(BisectionOutput) referenceOutput,
				(BisectionOutput) predictedOutput, lossWeight);
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		partialInference((BisectionModel) model, (BisectionInput) input,
				(BisectionOutput) partiallyLabeledOutput,
				(BisectionOutput) predictedOutput);
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

	@Override
	public double calculateSufferLoss(ExampleOutput correctOutput, ExampleOutput predictedOutput,
			PassiveAgressiveUpdate update) {
		throw new NotImplementedException();
	}

}
