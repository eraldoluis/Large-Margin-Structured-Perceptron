package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Arrays;
import java.util.Comparator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.bisection.BisectionOutput.WeightedPaper;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.maxbranching.KruskalAlgorithm;

/**
 * Prediction algorithm for bisection model. Based on Kruskal algorithm to
 * cluster candidate papers in confirmed and deleted sets. Then, use distance to
 * the artificial deleted node to rank papers within each cluster.
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
		mstAlgorithm = new KruskalAlgorithm(0, 2);
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
		mstAlgorithm.findMaxBranching(size, graph, output.getMst());

		// Compute confirmed papers from the found MST.
		output.computeSplitFromMst();

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
		mstAlgorithm.findMaxBranching(size, graph, predictedOutput.getMst());

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
		// Build graph to run MST algorithm.
		double[][] graph = new double[size][size];
		fillPartiaGraph(graph, model, input, partiallyLabeledOutput);

		// Guarantee that auxiliary data structures fit this instance.
		mstAlgorithm.realloc(size);
		// Find MST.
		mstAlgorithm.findMaxBranching(size, graph, predictedOutput.getMst());
		// Copy confirmed papers structure.
		predictedOutput.setConfirmedPapersEqualTo(partiallyLabeledOutput);
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
				int[] catBasFtrs = input.getBasicCategoricalFeatures(paper1,
						paper2);
				if (catBasFtrs == null) {
					// Skip inexistent edge.
					graph[paper1][paper2] = Double.NaN;
					continue;
				}

				// Categorical and numerical features of this edge.
				int[] ftrCodes = input.getFeatureCodes(paper1, paper2);
				double[] ftrValues = input.getFeatureValues(paper1, paper2);

				// Compute edge weight.
				double w = 0d;
				for (int idxFtr = 0; idxFtr < ftrCodes.length; ++idxFtr)
					w += model.getFeatureWeight(ftrCodes[idxFtr])
							* ftrValues[idxFtr];

				if (correct != null
						&& lossWeight != 0d
						&& (correct.isConfirmed(paper1) != correct
								.isConfirmed(paper2)))
					// Add margin for incorrect edges.
					w += lossWeight;

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

				// Check if edge exists in the input structure.
				int[] catBasFtrs = input.getBasicCategoricalFeatures(paper1,
						paper2);
				if (catBasFtrs == null) {
					// Inexistent edge.
					graph[paper1][paper2] = Double.NaN;
					continue;
				}

				// Categorical and numerical features of this edge.
				int[] ftrCodes = input.getFeatureCodes(paper1, paper2);
				double[] ftrValues = input.getFeatureValues(paper1, paper2);

				// Compute edge weight.
				double w = 0d;
				for (int idxFtr = 0; idxFtr < ftrCodes.length; ++idxFtr)
					w += model.getFeatureWeight(ftrCodes[idxFtr])
							* ftrValues[idxFtr];
				graph[paper1][paper2] = w;
			}
		}
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
			wPaper.confirmed = output.isConfirmed(paper);
			wPaper.weight = -graph[0][paper];
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

}
