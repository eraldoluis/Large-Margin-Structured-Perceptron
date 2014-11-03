package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive.PassiveAgressiveUpdate;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive.PredictionBasedUpdate;
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

	private ExecutorService executor;
	private int numberThreadsToFillWeight;

	private abstract class FillerWeights implements Callable<Integer>, Runnable {
		private int threadId;
		private int numberThreads;
		protected DPGSInput input;
		protected DPGSModel model;
		protected DPGSOutput correct;
		protected double lossWeight;
		private boolean loopUntilNumberTokens;

		public FillerWeights(int threadId, int numberThreads, DPGSInput input,
				DPGSModel model, DPGSOutput correct, double lossWeight,
				boolean loopUntilNumberTokens) {
			this.threadId = threadId;
			this.numberThreads = numberThreads;
			this.input = input;
			this.model = model;
			this.correct = correct;
			this.lossWeight = lossWeight;
			this.loopUntilNumberTokens = loopUntilNumberTokens;
		}

		private int hash(int idxHead, int idxModifier) {
			int hash = 1;
			hash = hash * 211 + idxHead;
			hash = hash * 421 + idxModifier;

			return hash;
		}

		protected abstract void fill(int numberTokens, int idxHead,
				int idxModifier, boolean loss);

		@Override
		public Integer call() throws Exception {
			int numTkns = input.size();
			boolean loss = (correct != null /* && lossWeight != 0d */);
			int numT = loopUntilNumberTokens ? numTkns : numTkns - 1;
			try {

				for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
					for (int idxModifier = 0; idxModifier <= numT; ++idxModifier) {
						if (hash(idxHead, idxModifier) % numberThreads == threadId) {
							fill(numTkns, idxHead, idxModifier, loss);
						}
					}
				}

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return null;
		}

		@Override
		public void run() {
			try {
				call();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	private class FillerSiblingWeights extends FillerWeights {

		public FillerSiblingWeights(int threadId, int numberThreads,
				DPGSInput input, DPGSModel model, DPGSOutput correct,
				double lossWeight) {
			super(threadId, numberThreads, input, model, correct, lossWeight,
					true);
		}

		@Override
		public Integer call() throws Exception {
			super.call();

			// TODO test (this factor should be generated).
			siblingsFactorWeights[0][0][0] = 0d;

			return null;
		}

		@Override
		public void run() {
			try {
				call();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		protected double getLossWeight(int idxHead, int idxModifier,
				int idxPreviousModifier, boolean loss) {

			if (!loss
					|| correct.isPreviousModifier(idxHead, idxModifier,
							idxPreviousModifier)) {
				return 0.0D;
			}

			return lossWeight;
		}

		@Override
		protected void fill(int numberTokens, int idxHead, int idxModifier,
				boolean loss) {
			/*
			 * Consider here the proper modifier pairs (that is both modifiers
			 * are real tokens/nodes) and pairs of the form <*, END>. Thus, do
			 * not consider <START, *> at this point. We deal with such pairs in
			 * the next block.
			 */
			double[] siblingsFactorWeightsHeadModifier = siblingsFactorWeights[idxHead][idxModifier];
			/*
			 * First modifier index depends whether the current modifier lies on
			 * the LEFT (then the first modifier index is '0') or on the RIGHT
			 * (then it is 'idxHead + 1') of the current head (idxHead).
			 */
			int firstModifier = (idxModifier <= idxHead ? 0 : idxHead + 1);

			for (int idxPreviousModifier = firstModifier; idxPreviousModifier < idxModifier; ++idxPreviousModifier) {
				int[] ftrs = input.getSiblingsFeatures(idxHead, idxModifier,
						idxPreviousModifier);
				if (ftrs != null)
					siblingsFactorWeightsHeadModifier[idxPreviousModifier] = model
							.getFeatureListScore(ftrs)
							+ getLossWeight(idxHead, idxModifier,
									idxPreviousModifier, loss);
				else
					siblingsFactorWeightsHeadModifier[idxPreviousModifier] = Double.NaN;
			}

			/*
			 * Modifier pairs of the form <START, *>. For modifiers on the left
			 * side of the current head, START is equal to 'idxHead'. While for
			 * modifiers on the right side of the current head, it is equal to
			 * 'numTkns'.
			 * 
			 * START index depends whether the current modifier lies on the LEFT
			 * side (then it is equal to 0) or on the RIGHT side (then it is
			 * numTkns) of the current head (idxHead).
			 */
			int idxSTART = (idxModifier <= idxHead ? idxHead : numberTokens);
			int[] ftrs = input.getSiblingsFeatures(idxHead, idxModifier,
					idxSTART);
			if (ftrs != null)
				siblingsFactorWeightsHeadModifier[idxSTART] = model
						.getFeatureListScore(ftrs)
						+ getLossWeight(idxHead, idxModifier, idxSTART, loss);
			else
				siblingsFactorWeightsHeadModifier[idxSTART] = Double.NaN;
		}
	}

	private class FillerGrandparentWeights extends FillerWeights {

		public FillerGrandparentWeights(int threadId, int numberThreads,
				DPGSInput input, DPGSModel model, DPGSOutput correct,
				double lossWeight) {
			super(threadId, numberThreads, input, model, correct, lossWeight,
					false);
		}

		protected double getLossWeight(int idxHead, int idxModifier,
				int idxGrandparent, boolean loss) {

			if (!loss
					|| (correct.getHead(idxHead) == idxGrandparent && correct
							.getHead(idxModifier) == idxHead)) {
				return 0.0D;
			}

			return lossWeight;
		}

		@Override
		protected void fill(int numberTokens, int idxHead, int idxModifier,
				boolean loss) {
			double[] grandparentFactorWeightsHeadModifier = grandparentFactorWeights[idxHead][idxModifier];

			// Fill factor weights for each grandparent.
			for (int idxGrandparent = 0; idxGrandparent < numberTokens; ++idxGrandparent) {
				// Get list of features for the current siblings
				// factor.
				int[] ftrs = input.getGrandparentFeatures(idxHead, idxModifier,
						idxGrandparent);
				// System.out.println(idxHead + " " + idxModifier + " " +
				// idxGrandparent);

				if (ftrs != null) {
					// Sum feature weights to achieve the factor
					// weight.
					grandparentFactorWeightsHeadModifier[idxGrandparent] = model
							.getFeatureListScore(ftrs);
					// Loss value for the current edge.
					grandparentFactorWeightsHeadModifier[idxGrandparent] += getLossWeight(
							idxHead, idxModifier, idxGrandparent, loss);
					;
				} else
					grandparentFactorWeightsHeadModifier[idxGrandparent] = Double.NaN;
			}

		}

	}

	private class FillerEdgeWeights extends FillerWeights {

		public FillerEdgeWeights(int threadId, int numberThreads,
				DPGSInput input, DPGSModel model, DPGSOutput correct,
				double lossWeight) {
			super(threadId, numberThreads, input, model, correct, lossWeight,
					false);
		}

		protected double getLossWeight(int idxHead, int idxModifier,
				boolean loss) {
			if (!loss || correct.getHead(idxModifier) == idxHead)
				return 0.0D;
			return lossWeight;
		}

		@Override
		protected void fill(int numberTokens, int idxHead, int idxModifier,
				boolean loss) {

			edgeFactorWeights[idxHead][idxModifier] = model
					.getFeatureListScore(input.getEdgeFeatures(idxHead,
							idxModifier))
					+ getLossWeight(idxHead, idxModifier, loss);
		}
	}

	/**
	 * Create a grandparent/sibling inference object that allocates the internal
	 * data structures to support the given maximum number of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public DPGSInference(int maxNumberOfTokens, int numberThreadsToFillWeight) {
		maxGSAlgorithm = new MaximumGrandparentSiblingsAlgorithm(
				maxNumberOfTokens);
		edgeFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];

		executor = Executors.newFixedThreadPool(numberThreadsToFillWeight);
		this.numberThreadsToFillWeight = numberThreadsToFillWeight;
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

		fillEdgeFactorWeights(model, input, null, 0d);
		fillGrandparentFactorWeights(model, input, null, 0d);
		fillSiblingsFactorWeights(model, input, null, 0d);

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
		fillEdgeFactorWeights(model, input, referenceOutput, lossWeight);

		lossWeight = 0.0d;

		fillGrandparentFactorWeights(model, input, referenceOutput, lossWeight);
		fillSiblingsFactorWeights(model, input, referenceOutput, lossWeight);

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
	public void fillEdgeFactorWeights(DPGSModel model, DPGSInput input,
			DPGSOutput correct, double lossWeight) {
		Collection<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>(
				numberThreadsToFillWeight);
		for (int i = 0; i < numberThreadsToFillWeight; i++) {
			tasks.add(new FillerEdgeWeights(i, numberThreadsToFillWeight,
					input, model, correct, lossWeight));
		}

		try {
			executor.invokeAll(tasks);
		} catch (InterruptedException e) { // TODO Auto-generatedcatch block
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// int numTkns = input.size();
		//
		// for (int idxHead = 0; idxHead < numTkns; ++idxHead)
		// for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier)
		// edgeFactorWeights[idxHead][idxModifier] = model
		// .getFeatureListScore(input.getEdgeFeatures(idxHead,
		// idxModifier));
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
		Collection<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>(
				numberThreadsToFillWeight);
		for (int i = 0; i < numberThreadsToFillWeight; i++) {
			tasks.add(new FillerGrandparentWeights(i,
					numberThreadsToFillWeight, input, model, correct,
					lossWeight));
		}

		try {
			executor.invokeAll(tasks);
		} catch (InterruptedException e) { // TODO Auto-generatedcatch block
			e.printStackTrace();
		}

		// // Loss augmented?
		// boolean loss = (correct != null && lossWeight != 0d);
		//
		// int numTkns = input.size();
		// for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
		// double[][] grandparentFactorWeightsHead =
		// grandparentFactorWeights[idxHead];
		// for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
		// double[] grandparentFactorWeightsHeadModifier =
		// grandparentFactorWeightsHead[idxModifier];
		// // Loss weight for the current edge (idxHead, idxModifier).
		// double lossWeightEdge = 0d;
		// if (loss && correct.getHead(idxModifier) != idxHead)
		// lossWeightEdge = lossWeight;
		// // Fill factor weights for each grandparent.
		// for (int idxGrandparent = 0; idxGrandparent < numTkns;
		// ++idxGrandparent) {
		// // Get list of features for the current siblings factor.
		// int[] ftrs = input.getGrandparentFeatures(idxHead,
		// idxModifier, idxGrandparent);
		// if (ftrs != null) {
		// // Sum feature weights to achieve the factor weight.
		// grandparentFactorWeightsHeadModifier[idxGrandparent] = model
		// .getFeatureListScore(ftrs);
		// // Loss value for the current edge.
		// grandparentFactorWeightsHeadModifier[idxGrandparent] +=
		// lossWeightEdge;
		// } else
		// grandparentFactorWeightsHeadModifier[idxGrandparent] = Double.NaN;
		// }
		// }
		// }
	}

	/**
	 * Fill the underlying weight of the siblings factors that are use by the
	 * dynamic programming algorithm.
	 * 
	 * @param model
	 * @param input
	 */
	private void fillSiblingsFactorWeights(DPGSModel model, DPGSInput input,
			DPGSOutput correct, double lossWeight) {
		Collection<Callable<Integer>> tasks = new ArrayList<Callable<Integer>>(
				numberThreadsToFillWeight);
		for (int i = 0; i < numberThreadsToFillWeight; i++) {
			tasks.add(new FillerSiblingWeights(i, numberThreadsToFillWeight,
					input, model, correct, lossWeight));
		}

		try {
			executor.invokeAll(tasks);
		} catch (InterruptedException e) { // TODO Auto-generatedcatch block
			e.printStackTrace();
		}

		// int numTkns = input.size();
		// for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
		// double[][] siblingsFactorWeightsHead =
		// siblingsFactorWeights[idxHead];
		//
		// /*
		// * Consider here the proper modifier pairs (that is both modifiers
		// * are real tokens/nodes) and pairs of the form <*, END>. Thus, do
		// * not consider <START, *> at this point. We deal with such pairs in
		// * the next block.
		// */
		// for (int idxModifier = 0; idxModifier <= numTkns; ++idxModifier) {
		// double[] siblingsFactorWeightsHeadModifier =
		// siblingsFactorWeightsHead[idxModifier];
		// /*
		// * First modifier index depends whether the current modifier
		// * lies on the LEFT (then the first modifier index is '0') or on
		// * the RIGHT (then it is 'idxHead + 1') of the current head
		// * (idxHead).
		// */
		// int firstModifier = (idxModifier <= idxHead ? 0 : idxHead + 1);
		// for (int idxPreviousModifier = firstModifier; idxPreviousModifier <
		// idxModifier; ++idxPreviousModifier) {
		// int[] ftrs = input.getSiblingsFeatures(idxHead,
		// idxModifier, idxPreviousModifier);
		// if (ftrs != null)
		// siblingsFactorWeightsHeadModifier[idxPreviousModifier] = model
		// .getFeatureListScore(ftrs);
		// else
		// siblingsFactorWeightsHeadModifier[idxPreviousModifier] = Double.NaN;
		// }
		//
		// /*
		// * Modifier pairs of the form <START, *>. For modifiers on the
		// * left side of the current head, START is equal to 'idxHead'.
		// * While for modifiers on the right side of the current head, it
		// * is equal to 'numTkns'.
		// *
		// * START index depends whether the current modifier lies on the
		// * LEFT side (then it is equal to 0) or on the RIGHT side (then
		// * it is numTkns) of the current head (idxHead).
		// */
		// int idxSTART = (idxModifier <= idxHead ? idxHead : numTkns);
		// int[] ftrs = input.getSiblingsFeatures(idxHead, idxModifier,
		// idxSTART);
		// if (ftrs != null)
		// siblingsFactorWeightsHeadModifier[idxSTART] = model
		// .getFeatureListScore(ftrs);
		// else
		// siblingsFactorWeightsHeadModifier[idxSTART] = Double.NaN;
		// }
		// }
		//
		// // TODO test (this factor should be generated).
		// siblingsFactorWeights[0][0][0] = 0d;
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
		DPGSInference inference = new DPGSInference(input.size(), 1);

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

	private double costP(DPGSOutput correctOutput, DPGSOutput predictedOutput) {
		double numberWrongEdges = 0.0d;
		int[] grandParentsCorrectOutput = correctOutput.getGrandparents();
		int[] grandParentsPredictedOutput = predictedOutput.getGrandparents();

		for (int idxHead = 0; idxHead < correctOutput.size(); ++idxHead) {
			if (grandParentsCorrectOutput[idxHead] != grandParentsPredictedOutput[idxHead])
				numberWrongEdges += 1.0d;
		}

		return numberWrongEdges;
	}

	@Override
	public double calculateSufferLoss(ExampleOutput correctOutput,
			ExampleOutput predictedOutput, PassiveAgressiveUpdate update) {

 		if (update == null) {
			update = new PredictionBasedUpdate();
		}

		DPGSOutput predicted = (DPGSOutput) update.getExampleOutput(
				correctOutput, predictedOutput);
		DPGSOutput correct = (DPGSOutput) correctOutput;

		double dif = .0d;
		/*
		 * For each head and modifier, check whether the predicted factor does
		 * not correspond to the correct one and, then, update the current model
		 * properly.
		 */
		int numTkns = correct.size();

		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			// Correct and predicted grandparent heads.
			int correctGrandparent = correct.getHead(idxHead);
			int predictedGrandparent = predicted.getGrandparent(idxHead);

			if (correctGrandparent != predictedGrandparent) {

				if (predictedGrandparent != -1)
					if (!Double
							.isNaN(edgeFactorWeights[predictedGrandparent][idxHead]))
						dif += edgeFactorWeights[predictedGrandparent][idxHead];

				if (correctGrandparent != -1)
					if (!Double
							.isNaN(edgeFactorWeights[correctGrandparent][idxHead]))
						dif -= edgeFactorWeights[correctGrandparent][idxHead];
			}

			/*
			 * Verify grandparent and siblings factors for differences between
			 * correct and predicted factors.
			 * 
			 * We start as previous token with the special 'idxHead' index is
			 * the index to indicate START and END tokens for LEFT modifiers.
			 * For RIGHT modifiers, we use the 'numTkns' index.
			 */
			int correctPreviousModifier = idxHead;
			int predictedPreviousModifier = idxHead;
			for (int idxModifier = 0; idxModifier <= numTkns; ++idxModifier) {
				// Is this token special (START or END).
				boolean isSpecialToken = (idxModifier == idxHead || idxModifier == numTkns);

				/*
				 * Is this modifier included in the correct or in the predicted
				 * structures for the current head or is it a special token.
				 * Special tokens are always present, by definition.
				 */
				boolean isCorrectModifier = (isSpecialToken || (correct
						.getHead(idxModifier) == idxHead));
				boolean isPredictedModifier = (isSpecialToken || predicted
						.isModifier(idxHead, idxModifier));

				if (!isCorrectModifier && !isPredictedModifier)
					/*
					 * Current modifier is neither included in the correct
					 * structure nor the predicted structure. Thus, skip it.
					 */
					continue;

				if (isCorrectModifier != isPredictedModifier) {
					//
					// Current modifier is misclassified.
					//

					if (isCorrectModifier) { // && !isPredictedModifier

						/*
						 * Current modifier is correct but the predicted
						 * structure does not set it as a modifier of the
						 * current head (false negative). Thus, increment the
						 * weight of both (grandparent and siblings) correct,
						 * but missed, factors.
						 */
						if (!Double
								.isNaN(siblingsFactorWeights[idxHead][idxModifier][correctPreviousModifier]))
							dif -= siblingsFactorWeights[idxHead][idxModifier][correctPreviousModifier];

						if (correctGrandparent != -1)
							if (!Double
									.isNaN(grandparentFactorWeights[idxHead][idxModifier][correctGrandparent]))
								dif -= grandparentFactorWeights[idxHead][idxModifier][correctGrandparent];
					} else { // !isCorrectModifier && isPredictedModifier

						/*
						 * Current modifier is not correct but the predicted
						 * structure does set it as a modifier of the current
						 * head (false positive). Thus, decrement the weight of
						 * both (grandparent and siblings) incorrectly predicted
						 * factors.
						 */
						if (!Double
								.isNaN(siblingsFactorWeights[idxHead][idxModifier][predictedPreviousModifier]))
							dif += siblingsFactorWeights[idxHead][idxModifier][predictedPreviousModifier];

						if (predictedGrandparent != -1)
							if (!Double
									.isNaN(grandparentFactorWeights[idxHead][idxModifier][predictedGrandparent]))
								dif += grandparentFactorWeights[idxHead][idxModifier][predictedGrandparent];
					}

				} else { // isCorrectModifier == isPredictedModifier
					/*
					 * The current modifier has been correctly predicted for the
					 * current head. Now, additionally check the previous
					 * modifier and the grandparent factor.
					 */

					if (correctPreviousModifier != predictedPreviousModifier) {

						/*
						 * Modifier is correctly predicted but previous modifier
						 * is NOT. Thus, the corresponding correct siblings
						 * factor is missing (false negative) and the predicted
						 * one is incorrectly predicted (false positive).
						 */
						if (!Double
								.isNaN(siblingsFactorWeights[idxHead][idxModifier][correctPreviousModifier]))
							dif -= siblingsFactorWeights[idxHead][idxModifier][correctPreviousModifier];

						if (!Double
								.isNaN(siblingsFactorWeights[idxHead][idxModifier][predictedPreviousModifier]))
							dif += siblingsFactorWeights[idxHead][idxModifier][predictedPreviousModifier];
					}

					if (!isSpecialToken
							&& correctGrandparent != predictedGrandparent) {
						/*
						 * Predicted modifier is correct but grandparent head is
						 * NOT. Thus, the corresponding correct grandparent
						 * factor is missing (false negative) and the predicted
						 * one is incorrectly predicted (false positive).
						 */
						if (correctGrandparent != -1)
							if (!Double
									.isNaN(grandparentFactorWeights[idxHead][idxModifier][correctGrandparent]))
								dif += -grandparentFactorWeights[idxHead][idxModifier][correctGrandparent];

						if (predictedGrandparent != -1)
							if (!Double
									.isNaN(grandparentFactorWeights[idxHead][idxModifier][predictedGrandparent]))
								dif += grandparentFactorWeights[idxHead][idxModifier][predictedGrandparent];
					}
				}

				if (isCorrectModifier) {
					// Update correct previous modifier.
					if (idxModifier == idxHead)
						/*
						 * Current token (idxToken) is the boundary token
						 * between left and right modifiers. Thus, the previous
						 * modifier for the next iteration is the special START
						 * token for right modifiers, that is 'numTkns'.
						 */
						correctPreviousModifier = numTkns;
					else
						correctPreviousModifier = idxModifier;
				}

				if (isPredictedModifier) {
					// Update predicted previous modifier.
					if (idxModifier == idxHead)
						/*
						 * Current token (idxToken) is the boundary token
						 * between left and right modifiers. Thus, the previous
						 * modifier for the next iteration is the special START
						 * token for right modifiers, that is 'numTkns'.
						 */
						predictedPreviousModifier = numTkns;
					else
						predictedPreviousModifier = idxModifier;
				}
			}
		}

		return dif + Math.sqrt(costP(correct, predicted));
	}

}
