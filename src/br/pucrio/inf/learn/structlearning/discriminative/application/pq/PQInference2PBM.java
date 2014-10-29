package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive.PassiveAgressiveUpdate;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.Quotation;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Prediction algorithm based on the perfect bipartite graph matching problem.
 * 
 * @author william
 * 
 */
public class PQInference2PBM implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((PQModel2) model, (PQInput2) input, (PQOutput2) output);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((PQModel2) model, (PQInput2) input,
				(PQOutput2) referenceOutput, (PQOutput2) predictedOutput,
				lossWeight);
	}

	public void inference(PQModel2 model, PQInput2 input, PQOutput2 output) {
		// Generate cost matrix
		double[][] costMatrix = generateCostMatrix(model, input, output, 0d);

		// Eliminate zeros and create artificial nodes from cost matrix
		double[][] hungarianCostMatrix = preProcessCostMatrix(costMatrix);

		// Run Hungarian Method in order to find Perfect Bipartite Matching
		ArrayList<int[]> P = PerfectBipartiteMatching
				.perfectBipartiteMatching(hungarianCostMatrix);

		// Eliminate negative cost arcs and arcs linked to artificial nodes
		postProcessPerfectMatching(costMatrix, hungarianCostMatrix, P, input);

		/*
		 * Initialize the output vector with zero, indicating the quotations are
		 * invalid. Then we assign Perfect Bipartite Matching solution to the
		 * output
		 */
		int[] edge;
		int pSize = P.size();
		int outputSize = output.size();
		for (int i = 0; i < outputSize; ++i)
			output.setAuthor(i, 0);
		for (int i = 0; i < pSize; ++i) {
			edge = P.get(i);
			output.setAuthor(edge[0], edge[1]);
		}
	}

	public void lossAugmentedInference(PQModel2 model, PQInput2 input,
			PQOutput2 referenceOutput, PQOutput2 predictedOutput,
			double lossWeight) {
		// Generate cost matrix
		double[][] costMatrix = generateCostMatrix(model, input,
				referenceOutput, lossWeight);

		// Eliminate zeros and create artificial nodes from cost matrix
		double[][] hungarianCostMatrix = preProcessCostMatrix(costMatrix);

		// Run Hungarian Method in order to find Perfect Bipartite Matching
		ArrayList<int[]> P = PerfectBipartiteMatching
				.perfectBipartiteMatching(hungarianCostMatrix);

		// Eliminate negative cost arcs and arcs linked to artificial nodes
		postProcessPerfectMatching(costMatrix, hungarianCostMatrix, P, input);

		// Initialize the output vector with zero, indicating the quotations
		// are invalid. Then we assign Perfect Bipartite Matching solution
		// to the output
		int[] edge;
		int pSize = P.size();
		int predictedOutputSize = predictedOutput.size();
		for (int i = 0; i < predictedOutputSize; ++i)
			predictedOutput.setAuthor(i, 0);
		for (int i = 0; i < pSize; ++i) {
			edge = P.get(i);
			predictedOutput.setAuthor(edge[0], edge[1]);
		}
	}

	public double[][] generateCostMatrix(PQModel2 model, PQInput2 input,
			PQOutput2 correctOutput, double lossWeight) {
		ArrayList<Integer> coreferences = new ArrayList<Integer>();
		Quotation[] quotationIndexes = input.getQuotationIndexes();

		// Create an ordered list of all the document coreferences. As there is
		// no nested
		// coreference, keep only the coreference start token.
		for (int i = 0; i < quotationIndexes.length; ++i) {
			int numberOfCoreferences = quotationIndexes[i]
					.getNumberOfCoreferences();

			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex = quotationIndexes[i]
						.getCoreferenceIndex(j);
				int coreferenceStart = coreferenceIndex[0];

				if (!coreferences.contains(coreferenceStart)) {
					// We search for a position to insert the new coreference in
					// the coreference list (O(n)). Then we insert the
					// coreference in
					// specified position.
					int insertionIndex = Collections.binarySearch(coreferences,
							coreferenceStart);
					if (insertionIndex < 0)
						coreferences.add(-insertionIndex - 1, coreferenceStart);
					else
						coreferences.add(insertionIndex + 1, coreferenceStart);
				}
			}
		}

		int totalOfCoreferences = coreferences.size();
		double[][] costMatrix = new double[quotationIndexes.length][totalOfCoreferences];

		// Fill the cost matrix with zero.
		for (int i = 0; i < quotationIndexes.length; ++i)
			for (int j = 0; j < totalOfCoreferences; ++j)
				costMatrix[i][j] = 0d;

		// Generate all costs for the cost matrix. Lines correspond to
		// quotations and columns correspond to coreferences.
		for (int i = 0; i < quotationIndexes.length; ++i) {
			int correctAuthor = 0;
			if (correctOutput != null)
				correctAuthor = correctOutput.getAuthor(i);

			int numberOfCoreferences = quotationIndexes[i]
					.getNumberOfCoreferences();

			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex = quotationIndexes[i]
						.getCoreferenceIndex(j);
				int coreferencePosition = coreferences
						.indexOf(coreferenceIndex[0]);

				double currentLoss = 0d;
				if (correctOutput != null)
					currentLoss = (j != correctAuthor ? lossWeight : 0d);

				double cost = 0d;
				int featureIndex;
				Iterator<Integer> it = input.getFeatureCodes(i, j).iterator();
				while (it.hasNext()) {
					featureIndex = it.next();
					cost += currentLoss + model.getFeatureWeight(featureIndex);
				}

				costMatrix[i][coreferencePosition] = cost;
			}
		}

		return costMatrix;
	}

	public double[][] preProcessCostMatrix(double[][] costMatrix) {
		int lineSize = costMatrix.length;
		int columnSize = (lineSize > 0 ? costMatrix[0].length : 0);
		int hcmSize = (lineSize > columnSize ? lineSize : columnSize);

		double[][] hungarianCostMatrix = new double[hcmSize][hcmSize];

		// Fill artificial arcs with zero and eliminate negative costs
		for (int i = 0; i < hcmSize; ++i)
			for (int j = 0; j < hcmSize; ++j)
				if (i < lineSize && j < columnSize) {
					if (costMatrix[i][j] > 0d)
						hungarianCostMatrix[i][j] = costMatrix[i][j];
					else
						hungarianCostMatrix[i][j] = 0d;
				} else
					hungarianCostMatrix[i][j] = 0d;

		return hungarianCostMatrix;
	}

	public void postProcessPerfectMatching(double[][] costMatrix,
			double[][] hungarianCostMatrix, ArrayList<int[]> P, PQInput2 input) {
		int[] edge;
		int lineSize = costMatrix.length;
		int columnSize = (lineSize > 0 ? costMatrix[0].length : 0);
		Quotation[] quotationIndexes = input.getQuotationIndexes();

		int pSize = P.size();
		for (int i = pSize - 1; i >= 0; --i) {
			edge = P.get(i);

			// Eliminate arcs linked to artificial nodes
			if (edge[0] >= lineSize)
				P.remove(i);
			else if (edge[1] >= columnSize)
				P.remove(i);
			// Eliminate negative cost arcs
			else if (costMatrix[edge[0]][edge[1]] < 0d)
				P.remove(i);
			// Eliminate arcs linked to coreferences that cannot be associated
			// with the
			// current quotation
			else if (edge[1] >= quotationIndexes[edge[0]]
					.getNumberOfCoreferences())
				P.remove(i);
		}
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
	
	@Override
	public double calculateSufferLoss(ExampleOutput correctOutput, ExampleOutput predictedOutput,
			PassiveAgressiveUpdate update) {
		throw new NotImplementedException();
	}

}
