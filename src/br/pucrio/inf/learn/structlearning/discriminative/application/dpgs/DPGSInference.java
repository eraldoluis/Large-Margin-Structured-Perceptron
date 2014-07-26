package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.gsmaxbranching.MaximumGrandparentSiblingsAlgorithm;

/**
 * Dependency parser with grandparent and siblings features.
 * 
 * This algorithm uses only a dynamic program that considers the sencod-order
 * features (grandparent and siblings), but the output can be an infeasible
 * parse that includes cycles.
 * 
 * @author eraldo
 * 
 */
public class DPGSInference implements Inference {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(DPGSInference.class);

	/**
	 * Dynamic programming algorithm to grandparent/siblings model.
	 */
	private MaximumGrandparentSiblingsAlgorithm maxGSAlgorithm;

	/**
	 * Edge factor weights for grandparent/siblings algorithm. The index for
	 * this array is (idxHead, idxModifier).
	 */
	private double[][] edgeFactorWeights;

	/**
	 * Grandparent factor weights for grandparent/siblings algorithm. The index
	 * for this array is (idxHead, idxModifier, idxGrandparent).
	 */
	private double[][][] grandparentFactorWeights;

	/**
	 * Siblings factor weights for grandparent/siblings algorithm. The index for
	 * this array is (idxHead, idxModifier, idxPreviousModifier). The indexes
	 * idxModifier and idxPreviousModifier can assume two special values:
	 * START/END for left modifiers and START/END for right modifiers. START is
	 * always assumed to be the first modifier of any sequence of siblings and
	 * END is always the last one. Since, for START and END nodes, there are
	 * only factors of the form (idxHead, idxModifier, START) and (idxHead, END,
	 * idxModifier), we can use the same index for START and END. The special
	 * index for START/END node on the left side of the head is 'idxHead'. We
	 * can use this index because no edge of the form (idxHead, idxHead) is
	 * allowed. The special node on the right side of the head is
	 * 'numberOfNodes', that is we create an additional position is every
	 * siblings array to store this special START/END node.
	 */
	private double[][][] siblingsFactorWeights;

	/**
	 * Whether to copy the grandparent/siblings prediction to the parse
	 * structure in the output.
	 */
	private boolean copyPredictionToParse;

	/**
	 * Create a grandparent/sibling inference object that allocates the internal
	 * data structures to support the given maximum number of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public DPGSInference(int maxNumberOfTokens) {
		maxGSAlgorithm = new MaximumGrandparentSiblingsAlgorithm(
				maxNumberOfTokens);
		edgeFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];
	}

	/**
	 * Set whether to copy the grandparent/siblings prediction to parse after
	 * inference.
	 * 
	 * @param val
	 */
	public void setCopyPredictionToParse(boolean val) {
		copyPredictionToParse = val;
	}

	/**
	 * Realloc the internal data structures to support the given maximum number
	 * of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public void realloc(int maxNumberOfTokens) {
		maxGSAlgorithm.realloc(maxNumberOfTokens);
		edgeFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];
	}

	/**
	 * Fill <code>output</code> with the maximum scoring structure according to
	 * the given input and model.
	 * 
	 * @param model
	 *            object with grandparent/siblings factor weights to be used.
	 * @param input
	 *            object with the input structure.
	 * @param output
	 *            output object to be filled with the maximum scoring output
	 *            structure.
	 */
	public void inference(DPGSModel model, DPGSInput input, DPGSOutput output) {
		// Generate inference problem for the given input.
		fillEdgeFactorWeights(model, input);
		fillGrandparentFactorWeights(model, input, null, 0d);
		fillSiblingsFactorWeights(model, input);

		// Solve the inference 2problem.
		double score = maxGSAlgorithm.findMaximumGrandparentSiblings(
				input.size(), edgeFactorWeights, grandparentFactorWeights,
				siblingsFactorWeights, null, null, output.getGrandparents(),
				output.getModifiers());

		LOG.debug(String.format("Solution score: %f", score));
		

		if (copyPredictionToParse)
			copyGrandparentToTree(output);
	}

	/**
	 * Fill <code>predictedOutput</code> with the maximum scoring structure
	 * according to the given input and model. Additionally, embed loss values
	 * in the objective function according to <code>referenceOutput</code>.
	 * 
	 * @param model
	 * @param input
	 * @param referenceOutput
	 * @param predictedOutput
	 * @param lossWeight
	 */
	public void lossAugmentedInference(DPGSModel model, DPGSInput input,
			DPGSOutput referenceOutput, DPGSOutput predictedOutput,
			double lossWeight) {
		// Generate loss-augmented inference problem for the given input.
		fillEdgeFactorWeights(model, input);
		fillGrandparentFactorWeights(model, input, referenceOutput, lossWeight);
		fillSiblingsFactorWeights(model, input);

		// Solve the inference problem.
		maxGSAlgorithm.findMaximumGrandparentSiblings(input.size(),
				edgeFactorWeights, grandparentFactorWeights,
				siblingsFactorWeights, null, null,
				predictedOutput.getGrandparents(),
				predictedOutput.getModifiers());
	}

	/**
	 * Fill the underlying weight of the edge factors that are use by the
	 * dynamic programming algorithm.
	 * 
	 * @param model
	 * @param input
	 */
	private void fillEdgeFactorWeights(DPGSModel model, DPGSInput input) {
		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead)
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier)
				edgeFactorWeights[idxHead][idxModifier] = model
						.getFeatureListScore(input.getEdgeFeatures(idxHead,
								idxModifier));
	}

	/**
	 * Fill the underlying weights of the grandparent factors that are used by
	 * the dynamic programming algorithm. Optionally, if <code>correct</code>
	 * and <code>lossWeight</code> are given, augment the factor weights with
	 * loss values.
	 * 
	 * @param model
	 * @param input
	 * @param correct
	 * @param lossWeight
	 */
	private void fillGrandparentFactorWeights(DPGSModel model, DPGSInput input,
			DPGSOutput correct, double lossWeight) {
		// Loss augmented?
		boolean loss = (correct != null && lossWeight != 0d);

		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			double[][] grandparentFactorWeightsHead = grandparentFactorWeights[idxHead];
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
				double[] grandparentFactorWeightsHeadModifier = grandparentFactorWeightsHead[idxModifier];
				// Loss weight for the current edge (idxHead, idxModifier).
				double lossWeightEdge = 0d;
				if (loss && correct.getHead(idxModifier) != idxHead)
					lossWeightEdge = lossWeight;
				// Fill factor weights for each grandparent.
				for (int idxGrandparent = 0; idxGrandparent < numTkns; ++idxGrandparent) {
					// Get list of features for the current siblings factor.
					int[] ftrs = input.getGrandparentFeatures(idxHead,
							idxModifier, idxGrandparent);
					if (ftrs != null) {
						// Sum feature weights to achieve the factor weight.
						grandparentFactorWeightsHeadModifier[idxGrandparent] = model
								.getFeatureListScore(ftrs);
						// Loss value for the current edge.
						grandparentFactorWeightsHeadModifier[idxGrandparent] += lossWeightEdge;
					} else
						grandparentFactorWeightsHeadModifier[idxGrandparent] = Double.NaN;
				}
			}
		}
	}

	/**
	 * Fill the underlying weight of the siblings factors that are use by the
	 * dynamic programming algorithm.
	 * 
	 * @param model
	 * @param input
	 */
	private void fillSiblingsFactorWeights(DPGSModel model, DPGSInput input) {
		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			double[][] siblingsFactorWeightsHead = siblingsFactorWeights[idxHead];

			/*
			 * Consider here the proper modifier pairs (that is both modifiers
			 * are real tokens/nodes) and pairs of the form <*, END>. Thus, do
			 * not consider <START, *> at this point. We deal with such pairs in
			 * the next block.
			 */
			for (int idxModifier = 0; idxModifier <= numTkns; ++idxModifier) {
				double[] siblingsFactorWeightsHeadModifier = siblingsFactorWeightsHead[idxModifier];
				/*
				 * First modifier index depends whether the current modifier
				 * lies on the LEFT (then the first modifier index is '0') or on
				 * the RIGHT (then it is 'idxHead + 1') of the current head
				 * (idxHead).
				 */
				int firstModifier = (idxModifier <= idxHead ? 0 : idxHead + 1);
				for (int idxPreviousModifier = firstModifier; idxPreviousModifier < idxModifier; ++idxPreviousModifier) {
					int[] ftrs = input.getSiblingsFeatures(idxHead,
							idxModifier, idxPreviousModifier);
					if (ftrs != null)
						siblingsFactorWeightsHeadModifier[idxPreviousModifier] = model
								.getFeatureListScore(ftrs);
					else
						siblingsFactorWeightsHeadModifier[idxPreviousModifier] = Double.NaN;
				}

				/*
				 * Modifier pairs of the form <START, *>. For modifiers on the
				 * left side of the current head, START is equal to 'idxHead'.
				 * While for modifiers on the right side of the current head, it
				 * is equal to 'numTkns'.
				 * 
				 * START index depends whether the current modifier lies on the
				 * LEFT side (then it is equal to 0) or on the RIGHT side (then
				 * it is numTkns) of the current head (idxHead).
				 */
				int idxSTART = (idxModifier <= idxHead ? idxHead : numTkns);
				int[] ftrs = input.getSiblingsFeatures(idxHead, idxModifier,
						idxSTART);
				if (ftrs != null)
					siblingsFactorWeightsHeadModifier[idxSTART] = model
							.getFeatureListScore(ftrs);
				else
					siblingsFactorWeightsHeadModifier[idxSTART] = Double.NaN;
			}
		}

		// TODO test (this factor should be generated).
		siblingsFactorWeights[0][0][0] = 0d;
	}

	/**
	 * Copy the grandparent structure to the parse tree in the given output. If
	 * the grandparent variable for a head is not defined (it is equal to -1),
	 * then use the modifier variables to choose a parent token.
	 * 
	 * 
	 * @param output
	 */
	private void copyGrandparentToTree(DPGSOutput output) {
		int numTkns = output.size();
		for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
			int head = output.getGrandparent(idxModifier);
			if (head < 0) {
				for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
					if (idxModifier == idxHead)
						continue;
					if (output.isModifier(idxHead, idxModifier)) {
						head = idxHead;
						break;
					}
				}
			}
			output.setHead(idxModifier, head);
		}
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((DPGSModel) model, (DPGSInput) input, (DPGSOutput) output);
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
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		partialInference((DPGSModel) model, (DPGSInput) input,
				(DPGSOutput) partiallyLabeledOutput,
				(DPGSOutput) predictedOutput);
		throw new NotImplementedException();
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

	/**
	 * Test code.
	 * 
	 * @param args
	 * @throws DPGSException
	 */
	public static void main(String[] args) throws DPGSException {
		int[][][] eFtrs = new int[5][5][];
		int[][][][] gFtrs = new int[5][5][5][];
		int[][][][] sFtrs = new int[5][6][6][];

		/*
		 * Grandparent degenerated features (root node and root children have no
		 * grandparent).
		 */
		// gFtrs[0][0][0] = new int[] { 0 };
		// gFtrs[0][1][0] = new int[] { 0 };
		// gFtrs[0][2][0] = new int[] { 0 };
		// gFtrs[0][3][0] = new int[] { 0 };
		// gFtrs[0][4][0] = new int[] { 0 };

		// Grandparent ordinary features.
		gFtrs[2][3][0] = new int[] { 10 };
		gFtrs[2][4][0] = new int[] { 10 };

		// Siblings degenerated features (leaf nodes have no modifier).
		sFtrs[0][0][0] = new int[] { 0 };
		sFtrs[0][5][5] = new int[] { 0 };
		sFtrs[1][1][1] = new int[] { 0 };
		sFtrs[1][5][5] = new int[] { 0 };
		sFtrs[2][2][2] = new int[] { 0 };
		sFtrs[2][5][5] = new int[] { 0 };
		sFtrs[3][3][3] = new int[] { 0 };
		sFtrs[3][5][5] = new int[] { 0 };
		sFtrs[4][4][4] = new int[] { 0 };
		sFtrs[4][5][5] = new int[] { 0 };

		// Siblings ordinary features.
		sFtrs[0][1][5] = new int[] { 10 };
		sFtrs[0][2][1] = new int[] { 10 };
		sFtrs[0][5][2] = new int[] { 10 };
		sFtrs[2][3][5] = new int[] { 10 };
		sFtrs[2][4][3] = new int[] { 10 };
		sFtrs[2][5][4] = new int[] { 10 };

		// Input structure.
		DPGSInput input = new DPGSInput(eFtrs, gFtrs, sFtrs);

		// Inference object.
		DPGSInference inference = new DPGSInference(input.size());

		// Model.
		DPGSModel model = new DPGSModel(0);

		// Feature weights.
		model.getParameters().put(0, new AveragedParameter(0d));
		model.getParameters().put(10, new AveragedParameter(10d));

		// Output to be filled.
		DPGSOutput output = input.createOutput();

		// Inference.
		inference.inference(model, input, output);

		// Print output.
		System.out.println(output);
	}
}
