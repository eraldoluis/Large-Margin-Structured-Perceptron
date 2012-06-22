package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.gsmaxbranching.MaximumGrandparentSiblingsAlgorithm;
import br.pucrio.inf.learn.util.maxbranching.MaximumBranchingAlgorithm;

/**
 * Dependency parser with grandparent and siblings features based on dual
 * decomposition as in Koo et al. (EMNLP-2010).
 * 
 * This algorithm uses two sub-solvers: maximum branching to find a rooted tree
 * and a dynamic program that considers the sencod-order features (grandparent
 * and siblings). The dual decomposition, through Lagrangean relaxation,
 * guarantees that the solution of the later solver respects rooted tree
 * constraints with no need to directly consider these constraints in the
 * dynamic program, which would be very difficult, since that is a NP-hard
 * problem.
 * 
 * @author eraldo
 * 
 */
public class DPGSInference implements Inference {

	/**
	 * Log object.
	 */
	private final static Log LOG = LogFactory.getLog(DPGSInference.class);

	/**
	 * Special root node.
	 */
	private int root;

	/**
	 * Chu-Liu-Edmonds algoritmo to maximum branching.
	 */
	private MaximumBranchingAlgorithm maxBranchAlgorithm;

	/**
	 * Dynamic programming algorithm to grandparent/siblings model.
	 */
	private MaximumGrandparentSiblingsAlgorithm maxGSAlgorithm;

	/**
	 * Graph weights for the maximum branching algorithm. Nodes are tokens. The
	 * index for this array is (idxHead, idxModifier).
	 */
	private double[][] graph;

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
	 * Dual variables for grandparent constraints. The index for this array is
	 * (idxHead, idxModifier).
	 */
	private double[][] dualGrandparentVariables;

	/**
	 * Dual variables for modifier constraints. The index for this array is
	 * (idxHead, idxModifier).
	 */
	private double[][] dualModifierVariables;

	/**
	 * Maximum number of subgradient descent steps.
	 */
	private int maxNumberOfSubgradientSteps;

	/**
	 * Create a grandparent/sibling inference object that allocates the internal
	 * data structures to support the given maximum number of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public DPGSInference(int root, int maxNumberOfTokens) {
		this.root = root;
		maxBranchAlgorithm = new MaximumBranchingAlgorithm(maxNumberOfTokens);
		maxGSAlgorithm = new MaximumGrandparentSiblingsAlgorithm(
				maxNumberOfTokens);
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];
		dualGrandparentVariables = new double[maxNumberOfTokens][maxNumberOfTokens];
		dualModifierVariables = new double[maxNumberOfTokens][maxNumberOfTokens];
		maxNumberOfSubgradientSteps = 2;
	}

	/**
	 * Realloc the internal data structures to support the given maximum number
	 * of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public void realloc(int maxNumberOfTokens) {
		maxBranchAlgorithm.realloc(maxNumberOfTokens);
		maxGSAlgorithm.realloc(maxNumberOfTokens);
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];
		dualGrandparentVariables = new double[maxNumberOfTokens][maxNumberOfTokens];
		dualModifierVariables = new double[maxNumberOfTokens][maxNumberOfTokens];

		// TODO test
		maxBranchAlgorithm.setOnlyPositiveEdges(true);
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
		fillGraph(model, input);
		fillGrandparentFactorWeights(model, input, null, 0d);
		fillSiblingsFactorWeights(model, input);

		// Solve the inference problem.
		subgradientMethod(input, output);
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
		// Clear dual variables.
		clearDualVars(input.size());

		// Generate loss-augmented inference problem for the given input.
		fillGraph(model, input);
		fillGrandparentFactorWeights(model, input, referenceOutput, lossWeight);
		fillSiblingsFactorWeights(model, input);

		// Solve the inference problem.
		subgradientMethod(input, predictedOutput);
	}

	/**
	 * Using the underlying graph weights and grandparent/siblings factor
	 * weights, build the maximum scoring output structure and store in
	 * <code>output</code>.
	 * 
	 * @param input
	 * @param output
	 */
	private void subgradientMethod(DPGSInput input, DPGSOutput output) {
		/*
		 * Sparse arrays of updates for dual variables per head token in each
		 * iteration. They are used to update the proper dual variable arrays
		 * and the factor weight arrays in an effient way after each iteration.
		 */
		HashMap<Integer, HashMap<Integer, Double>> dualGrandparentDelta = new HashMap<Integer, HashMap<Integer, Double>>();
		HashMap<Integer, HashMap<Integer, Double>> dualModifierDelta = new HashMap<Integer, HashMap<Integer, Double>>();

		double prevDualObjectiveValue = Double.NaN;
		int numTkns = input.size();
		// Number of times that the dual objective has been incremented.
		int numDualObjectiveIncrements = 0;
		double[] dualObjectiveValues = new double[numTkns];
		double lambda = Double.NaN;
		for (int step = 0; step < maxNumberOfSubgradientSteps; ++step) {
			// Value of the dual objective function in this step.
			double dualObjectiveValue = 0d;

			if (step == 0) {
				// Fill the maximum branching for the current dual values.
				dualObjectiveValue = maxBranchAlgorithm.findMaxBranching(
						numTkns, graph, output.getHeads());

				// Fill the complete maximum grandparent/siblings structure.
				for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
					dualObjectiveValues[idxHead] = maxGSAlgorithm
							.findMaximumGrandparentSiblingsForHead(numTkns,
									idxHead, grandparentFactorWeights[idxHead],
									siblingsFactorWeights[idxHead],
									dualGrandparentVariables,
									dualModifierVariables,
									output.getGrandparents(),
									output.getModifiers());
					dualObjectiveValue += dualObjectiveValues[idxHead];
				}

				/*
				 * The first step size is equal to the difference between the
				 * weights of the grandparent/siblings structure and the parse
				 * structure under the grandparent/siblings objective function.
				 */
				lambda = dualObjectiveValue
						- maxGSAlgorithm
								.calcObjectiveValueOfParse(output,
										grandparentFactorWeights,
										siblingsFactorWeights,
										dualGrandparentVariables,
										dualModifierVariables);
			} else {
				/*
				 * Set of heads whose dual vars were updated in the previous
				 * iteration. We only need to recalculate the GS structure for
				 * these heads.
				 */
				HashSet<Integer> updatedHeads = new HashSet<Integer>();

				// Update grandparent dual variables.
				for (Entry<Integer, HashMap<Integer, Double>> headEntry : dualGrandparentDelta
						.entrySet()) {
					int idxHead = headEntry.getKey();
					for (Entry<Integer, Double> entryModifier : headEntry
							.getValue().entrySet()) {
						int idxModifier = entryModifier.getKey();

						/*
						 * GS structure must be recalculated for this modifier
						 * index due to this grandparent dual variable update.
						 */
						updatedHeads.add(idxModifier);

						double inc = entryModifier.getValue();
						dualGrandparentVariables[idxHead][idxModifier] += inc;
						graph[idxHead][idxModifier] += inc;
					}
				}

				// Update modifier dual variables.
				for (Entry<Integer, HashMap<Integer, Double>> headEntry : dualModifierDelta
						.entrySet()) {
					int idxHead = headEntry.getKey();

					/*
					 * GS structure must be recalculated for this head due to
					 * this modifier dual variable update.
					 */
					updatedHeads.add(idxHead);

					for (Entry<Integer, Double> entryModifier : headEntry
							.getValue().entrySet()) {
						int idxModifier = entryModifier.getKey();
						double inc = entryModifier.getValue();
						dualModifierVariables[idxHead][idxModifier] += inc;
						graph[idxHead][idxModifier] += inc;
					}
				}

				// Clear updated heads.
				dualGrandparentDelta.clear();
				dualModifierDelta.clear();

				// Fill the maximum branching for the current dual values.
				dualObjectiveValue = maxBranchAlgorithm.findMaxBranching(
						numTkns, graph, output.getHeads());

				for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
					if (updatedHeads.contains(idxHead))
						// Recompute GS structure only for updated heads.
						dualObjectiveValues[idxHead] = maxGSAlgorithm
								.findMaximumGrandparentSiblingsForHead(numTkns,
										idxHead,
										grandparentFactorWeights[idxHead],
										siblingsFactorWeights[idxHead],
										dualGrandparentVariables,
										dualModifierVariables,
										output.getGrandparents(),
										output.getModifiers());

					dualObjectiveValue += dualObjectiveValues[idxHead];
				}

				if (dualObjectiveValue > prevDualObjectiveValue)
					/*
					 * Decrement step size whenever the dual function increases.
					 * An increase in the dual function means that the algorithm
					 * passed over the optimum. Thus, the step size is too
					 * large.
					 */
					++numDualObjectiveIncrements;
			}

			// Update previous solution weight.
			prevDualObjectiveValue = dualObjectiveValue;

			// Step size.
			double stepSize = lambda / (1 + numDualObjectiveIncrements);

			// double stepSize = lambda;
			// if (numDualObjectiveIncrements > 2)
			// stepSize /= Math.log(1 + numDualObjectiveIncrements)
			// / Math.log(2);

			// Compute updates on dual variables.
			for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
				HashMap<Integer, Double> dualGrandparentDeltaOfHead = null;
				HashMap<Integer, Double> dualSiblingsDeltaOfHead = null;
				for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
					/*
					 * Get value of edge variable (idxHead, idxModifier) in the
					 * three set of variables: parse, grandparent and siblings.
					 */
					boolean isBranching = (output.getHead(idxModifier) == idxHead);
					boolean isGrandparent = (output.getGrandparent(idxModifier) == idxHead);
					boolean isModifier = output
							.isModifier(idxHead, idxModifier);

					if (isGrandparent != isBranching) {
						// Grandparent variable differs from parse.
						if (dualGrandparentDeltaOfHead == null) {
							dualGrandparentDeltaOfHead = new HashMap<Integer, Double>();
							dualGrandparentDelta.put(idxHead,
									dualGrandparentDeltaOfHead);
						}

						if (isBranching)
							incrementeSparseArrayItem(
									dualGrandparentDeltaOfHead, idxModifier,
									-stepSize);
						else
							incrementeSparseArrayItem(
									dualGrandparentDeltaOfHead, idxModifier,
									stepSize);
					}

					if (isModifier != isBranching) {
						// Modifier variable differs from parse.
						if (dualSiblingsDeltaOfHead == null) {
							dualSiblingsDeltaOfHead = new HashMap<Integer, Double>();
							dualModifierDelta.put(idxHead,
									dualSiblingsDeltaOfHead);
						}

						if (isBranching)
							incrementeSparseArrayItem(dualSiblingsDeltaOfHead,
									idxModifier, -stepSize);
						else
							incrementeSparseArrayItem(dualSiblingsDeltaOfHead,
									idxModifier, stepSize);
					}
				}
			}

			if (dualGrandparentDelta.size() == 0
					&& dualModifierDelta.size() == 0) {
				LOG.info("Optimum found at step " + step + " after "
						+ numDualObjectiveIncrements
						+ " dual objective increments.");
				// Stop if the optimality condition is reached.
				break;
			}
		}

		double objectiveValue = 0d;
		for (int idxHead = 0; idxHead < numTkns; ++idxHead)
			objectiveValue += dualObjectiveValues[idxHead];
		LOG.info(String.format(
				"Dual GS objective value is %f for GS structure.",
				objectiveValue));
		LOG.info(String.format(
				"Dual GS objective value is %f for parse structure.",
				maxGSAlgorithm.calcObjectiveValueOfParse(output,
						grandparentFactorWeights, siblingsFactorWeights,
						dualGrandparentVariables, dualModifierVariables)));
	}

	/**
	 * Utilitary method to increment a dictionary item. If the item is not
	 * present in the dictionary yet, then creates a new entry.
	 * 
	 * @param array
	 * @param key
	 * @param val
	 */
	private void incrementeSparseArrayItem(HashMap<Integer, Double> array,
			int key, double val) {
		Double oldVal = array.get(key);
		if (oldVal == null)
			array.put(key, val);
		else
			array.put(key, oldVal + val);
	}

	/**
	 * Fill the underlying graph with the values of the underlying dual
	 * variables.
	 * 
	 * @param model
	 * @param input
	 */
	private void fillGraph(DPGSModel model, DPGSInput input) {
		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier)
				graph[idxHead][idxModifier] = dualGrandparentVariables[idxHead][idxModifier]
						+ dualModifierVariables[idxHead][idxModifier];
			if (root >= 0)
				graph[idxHead][root] = Double.NaN;
		}
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
		// Is loss weight used?
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
			}

			/*
			 * Modifier pairs of the form <START, *>. For modifiers on the left
			 * side of the current head, START is equal to 'idxHead'. While for
			 * modifiers on the right side of the current head, it is equal to
			 * 'numTkns'.
			 */
			for (int idxModifier = 0; idxModifier <= numTkns; ++idxModifier) {
				double[] siblingsFactorWeightsHeadModifier = siblingsFactorWeightsHead[idxModifier];
				/*
				 * START index depends whether the current modifier lies on the
				 * LEFT side (then it is equal to 0) or on the RIGHT side (then
				 * it is numTkns) of the current head (idxHead).
				 */
				int start = (idxModifier <= idxHead ? idxHead : numTkns);
				int[] ftrs = input.getSiblingsFeatures(idxHead, idxModifier,
						start);
				if (ftrs != null)
					siblingsFactorWeightsHeadModifier[start] = model
							.getFeatureListScore(ftrs);
				else
					siblingsFactorWeightsHeadModifier[start] = Double.NaN;
			}
		}
	}

	/**
	 * Clear the values of the dual variables.
	 * 
	 * @param numberOfTokens
	 */
	private void clearDualVars(int numberOfTokens) {
		for (int idxHead = 0; idxHead < numberOfTokens; ++idxHead) {
			Arrays.fill(dualGrandparentVariables[idxHead], 0, numberOfTokens,
					0d);
			Arrays.fill(dualModifierVariables[idxHead], 0, numberOfTokens, 0d);
		}
	}

	/**
	 * Return the maximum number of steps that the subgradient method can
	 * perform before returning. That method can still return before whenever it
	 * finds the optimum solution.
	 * 
	 * @return
	 */
	public int getMaxNumberOfSubgradientSteps() {
		return maxNumberOfSubgradientSteps;
	}

	/**
	 * Set the maximum number of steps that the subgradient method can perform
	 * before returning. That method can still return before whenever it finds
	 * the optimum solution.
	 * 
	 * @param maxNumberOfSubgradientSteps
	 */
	public void setMaxNumberOfSubgradientSteps(int maxNumberOfSubgradientSteps) {
		this.maxNumberOfSubgradientSteps = maxNumberOfSubgradientSteps;
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
		DPGSInput input = new DPGSInput(gFtrs, sFtrs);

		// Inference object.
		DPGSInference inference = new DPGSInference(0, input.size());

		// Model.
		DPGSModel model = new DPGSModel(0);

		// Feature weights.
		model.getParameters().put(0, new AveragedParameter(0d));
		model.getParameters().put(10, new AveragedParameter(10d));

		// Output to be filled.
		DPGSOutput output = input.createOutput();

		// Inference.
		inference.setMaxNumberOfSubgradientSteps(100000);
		inference.inference(model, input, output);

		// Print output.
		System.out.println(output);
	}
}
