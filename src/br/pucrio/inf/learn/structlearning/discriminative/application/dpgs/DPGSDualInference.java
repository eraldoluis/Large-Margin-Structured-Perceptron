package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.gsmaxbranching.MaximumGrandparentSiblingsAlgorithm;
import br.pucrio.inf.learn.util.maxbranching.DirectedMaxBranchAlgorithm;

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
public class DPGSDualInference implements Inference {
	private static int a = 1;
	/**
	 * Log object.
	 */
	private final static Log LOG = LogFactory.getLog(DPGSDualInference.class);

	/**
	 * Chu-Liu-Edmonds algoritmo to maximum branching.
	 */
	private DirectedMaxBranchAlgorithm maxBranchAlgorithm;

	/**
	 * Dynamic programming algorithm to grandparent/siblings model.
	 */
	private MaximumGrandparentSiblingsAlgorithm maxGSAlgorithm;

	/**
	 * Parameter between 0 and 1 that indicates the fraction of edge factor
	 * weights that are passed to the maximum branching problem. The remaining
	 * weight is left to the grandparent/siblings algorithm.
	 */
	private double beta;

	/**
	 * Graph weights for the maximum branching algorithm. Nodes are tokens. The
	 * index for this array is (idxHead, idxModifier).
	 */
	private double[][] graph;

	/**
	 * Edge factor weights. The index for this array is (idxHead, idxModifier).
	 * A fraction of these weights (given by <code>beta</code>) is passed to the
	 * maximum branching algorithm and the remaining (<code>1 - beta</code>)
	 * goes to the grandparent/siblings algorithm.
	 */
	private double[][] edgeFactorWeights;

	/**
	 * Grandparent factor weights for grandparent/siblings algorithm. The index
	 * for this array is (idxHead, idxModifier, idxGrandparent).
	 */
	private double[][][] grandparentFactorWeights;

	/**
	 * Siblings factor weight vector for grandparent/siblings algorithm. The
	 * index for this array is (idxHead, idxModifier, idxPreviousModifier). <br>
	 * <br>
	 * The indexes idxModifier and idxPreviousModifier can assume two special
	 * values: START/END for left modifiers and START/END for right modifiers.
	 * START is always assumed to be the first modifier of any sequence of
	 * siblings and END is always the last one. For instance, consider that the
	 * head token 5 has two modifiers, namely tokens 7 and 8. Hence, the
	 * following siblings factors are present: (5, 7, START), (5, 8, 7) and (5,
	 * END, 8). <br>
	 * <br>
	 * Given a siblings factor (head, modifier, prevModifier), START special
	 * nodes can only occur as prevModifier, while END special nodes can only
	 * occur as modifier. Therefore, we can use the same index in the
	 * <code>siblingsFactorWeights</code> vector to represent START and END
	 * special nodes. <br>
	 * <br>
	 * The special index for START/END nodes that occur on the left side of
	 * (smaller than) the head is 'idxHead'. We can use this index because no
	 * edge of the form (idxHead, idxHead) is allowed. <br>
	 * <br>
	 * The special nodes that occur on the right side of (grater than) the head
	 * is 'numberOfNodes'. Hence, we create an additional position in every
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

	private int numPredictions;

	private int numSubGradSteps;

	/**
	 * Create a grandparent/sibling inference object that allocates the internal
	 * data structures to support the given maximum number of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public DPGSDualInference(int maxNumberOfTokens) {
		// Fraction of the edge factor weights used in the maxbranch alg.
		this.beta = 0.5d;
		// Maximum branching algorithm.
		maxBranchAlgorithm = new DirectedMaxBranchAlgorithm(maxNumberOfTokens);
		maxBranchAlgorithm.setCheckUniqueRoot(false);
		// Grandparent/siblings algorithm.
		maxGSAlgorithm = new MaximumGrandparentSiblingsAlgorithm(
				maxNumberOfTokens);
		// Branching weights.
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
		// Factor weights.
		edgeFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];
		// Dual variables.
		dualGrandparentVariables = new double[maxNumberOfTokens][maxNumberOfTokens];
		dualModifierVariables = new double[maxNumberOfTokens][maxNumberOfTokens];

		maxNumberOfSubgradientSteps = 2;
		maxBranchAlgorithm.setOnlyPositiveEdges(false);
	}

	/**
	 * Realloc the internal data structures to support the given maximum number
	 * of tokens.
	 * 
	 * @param maxNumberOfTokens
	 */
	public void realloc(int maxNumberOfTokens) {
		// Maximum branching algorithm.
		maxBranchAlgorithm.realloc(maxNumberOfTokens);
		// Grandparent/siblings algorithm.
		maxGSAlgorithm.realloc(maxNumberOfTokens);
		// Maximum branching graph weights.
		graph = new double[maxNumberOfTokens][maxNumberOfTokens];
		// Factor weights.
		edgeFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens];
		grandparentFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens][maxNumberOfTokens];
		siblingsFactorWeights = new double[maxNumberOfTokens][maxNumberOfTokens + 1][maxNumberOfTokens + 1];
		// Dual variables.
		dualGrandparentVariables = new double[maxNumberOfTokens][maxNumberOfTokens];
		dualModifierVariables = new double[maxNumberOfTokens][maxNumberOfTokens];
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
		// Clear dual variables.
		clearDualVars(input.size());

		// Generate inference problem for the given input.
		fillEdgeFactorWeights(model, input);
		fillGraph(input.size());
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
		fillEdgeFactorWeights(model, input);
		fillGraph(input.size());
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
		// Number of predictions performed.
		++numPredictions;

		// Dual objetive function value in the previous iteration.
		double prevDualObjectiveValue = Double.NaN;
		// Number of tokens in the current input.
		int numTkns = input.size();
		// Number of times that the dual objective has been incremented.
		int numDualObjectiveIncrements = 0;
		// Dual objective function values for each head.
		double[] dualObjectiveValues = new double[numTkns];
		// Initial step size.
		double lambda = Double.NaN;

		// Fill the maximum branching for a zero-weight graph.
		double dualObjectiveValue = maxBranchAlgorithm.findMaxBranching(
				numTkns, graph, output.getHeads());
		
		double gsWeight = 0;
		// Fill the complete maximum grandparent/siblings structure.
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			dualObjectiveValues[idxHead] = maxGSAlgorithm
					.findMaximumGrandparentSiblingsForHead(numTkns, idxHead,
							edgeFactorWeights,
							grandparentFactorWeights[idxHead],
							siblingsFactorWeights[idxHead],
							dualGrandparentVariables, dualModifierVariables,
							output.getGrandparents(), output.getModifiers());
			gsWeight += dualObjectiveValues[idxHead];
			dualObjectiveValue += dualObjectiveValues[idxHead];
		}

		// Current best solution.
		
		double treeWeight = maxGSAlgorithm.calcObjectiveValueOfParse(
				output.getHeads(), output.size(), edgeFactorWeights,
				grandparentFactorWeights, siblingsFactorWeights, null, null);
		
		double bestOutputWeight = treeWeight;
		int[] bestOutput = output.getHeads().clone();

		/*
		 * The first step size is equal to the difference between the weights of
		 * the grandparent/siblings structure and the parse structure under the
		 * grandparent/siblings objective function.
		 */
		lambda = gsWeight - treeWeight;

		LOG.info("input " + a++ + " lambda " + lambda);
		
		if (lambda == 0d)
			lambda = 1d;
		

		/*
		 * Set of heads whose dual vars were updated in the previous iteration.
		 * We only need to recalculate the GS structure for these heads.
		 */
		boolean[] updatedHeads = new boolean[numTkns];
		int step;
		double outputWeight = treeWeight;
		
		for (step = 0; step < maxNumberOfSubgradientSteps; ++step) {

			// Number of subgradient steps performed.
			++numSubGradSteps;

			// Step size.
			double stepSize = lambda / (1 + numDualObjectiveIncrements);
			
			//LOG.info("stepsize " + stepSize);
			
			// Update dual variables.
			boolean updated = false;
			for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
				for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
					/*
					 * Get value of edge variable (idxHead, idxModifier) in the
					 * three set of variables: parse, grandparent and siblings.
					 */
					boolean isBranching = (output.getHead(idxModifier) == idxHead);
					boolean isGrandparent = (output.getGrandparent(idxModifier) == idxHead);
					boolean isModifier = output
							.isModifier(idxHead, idxModifier);

					// Grandparent variable differs from parse.
					if (isGrandparent != isBranching) {
						// The subproblem for idxModifier head token changed.
						updatedHeads[idxModifier] = true;
						updated = true;
						
						if (isBranching)
							dualGrandparentVariables[idxHead][idxModifier] -= stepSize;
						else
							dualGrandparentVariables[idxHead][idxModifier] += stepSize;
					}

					// Modifier variable differs from parse.
					if (isModifier != isBranching) {
						// The subproblem for idxHead head token changed.
						updatedHeads[idxHead] = true;
						updated = true;
						
						if (isBranching)
							dualModifierVariables[idxHead][idxModifier] -= stepSize;
						else
							dualModifierVariables[idxHead][idxModifier] += stepSize;
					}
				}
			}
			
			if (!updated) {
				LOG.info(String
						.format("Optimum found at step %d after %d dual objective increments. Dual objective: %f. Weight: %f",
								step, numDualObjectiveIncrements,
								dualObjectiveValue, maxGSAlgorithm
										.calcObjectiveValueOfParse(
												output.getHeads(),
												output.size(),
												edgeFactorWeights,
												grandparentFactorWeights,
												siblingsFactorWeights, null,
												null)));
				// LOG.info("\n" + output.toString());

				// Stop if the optimality condition is reached.
				break;
			} else {
				/*LOG.info(String
						.format("Solution at step %d. Dual objective: %f. Last Output Weight: %f. Best Output Weight %f.(Dual objective - Best Output) %f ",
						step,dualObjectiveValue,outputWeight,bestOutputWeight, dualObjectiveValue - bestOutputWeight));
				*/
				/*LOG.info(String
				.format("Solution at step %d after %d dual objective increments. Dual objective: %f. Weight: %f",
				step, numDualObjectiveIncrements,
				dualObjectiveValue, maxGSAlgorithm
				.calcObjectiveValueOfParse(
				output.getHeads(),
				output.size(),
				edgeFactorWeights,
				grandparentFactorWeights,
				siblingsFactorWeights, null,
				null)));*/
				//LOG.info("\n" + output.toString());
			}

			// Value of the dual objective function in this step.
			dualObjectiveValue = 0d;

			// Fill graph for the maximum branching problem using the dual vars.
			fillGraph(numTkns);

			// Fill the maximum branching for the current dual values.
			dualObjectiveValue = maxBranchAlgorithm.findMaxBranching(numTkns,
					graph, output.getHeads());

			// Update the best output up to this iteration.
			outputWeight = maxGSAlgorithm
					.calcObjectiveValueOfParse(output.getHeads(), numTkns,
							edgeFactorWeights, grandparentFactorWeights,
							siblingsFactorWeights, null, null);
			
			if (outputWeight > bestOutputWeight) {
				bestOutputWeight = outputWeight;
				for (int tkn = 0; tkn < numTkns; ++tkn)
					bestOutput[tkn] = output.getHead(tkn);
			}

			/*
			 * Compute the best GS structure for heads whose dual variables have
			 * been updated.
			 */
			for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
				if (updatedHeads[idxHead])
					dualObjectiveValues[idxHead] = maxGSAlgorithm
							.findMaximumGrandparentSiblingsForHead(numTkns,
									idxHead, edgeFactorWeights,
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
				 * Decrement step size whenever the dual function increases. An
				 * increase in the dual function means that the algorithm passed
				 * over the optimum. Thus, the step size is too large.
				 */
				++numDualObjectiveIncrements;

			// Store the dual objective value for the previous solution.
			prevDualObjectiveValue = dualObjectiveValue;

			// Clear flag array of updated heads for the next iteration.
			Arrays.fill(updatedHeads, false);
		}
		
		LOG.info(String
				.format("Stop in step %d with Dual objective: %f and Weight: %f",
						step,
						dualObjectiveValue, maxGSAlgorithm
								.calcObjectiveValueOfParse(
										output.getHeads(),
										output.size(),
										edgeFactorWeights,
										grandparentFactorWeights,
										siblingsFactorWeights, null,
										null)));

		// Copy the best parse tree to the output structure.
		for (int tkn = 0; tkn < numTkns; ++tkn)
			output.setHead(tkn, bestOutput[tkn]);
	}

	public static void printDualVars(int numTkns, double[][] vars) {
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			System.out.print("Head " + idxHead + ":");
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
				System.out.print("\t" + idxModifier + "="
						+ vars[idxHead][idxModifier]);
			}
			System.out.println();
		}
	}
	
	private double convertNan(double a){
		if(Double.isNaN(a))
			return 0.0d;
		
		return a;
	}

	/**
	 * Fill the underlying weights of the edge factors that are used by the
	 * optimization algorithms.
	 * 
	 * @param model
	 * @param input
	 */
	private void fillEdgeFactorWeights(DPGSModel model, DPGSInput input) {
		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead)
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier){
				int[] ftrs = input.getEdgeFeatures(idxHead,idxModifier);
				
 				if (ftrs != null)
					edgeFactorWeights[idxHead][idxModifier] = convertNan(model
							.getFeatureListScore(ftrs));
				else
					edgeFactorWeights[idxHead][idxModifier] = convertNan(Double.NaN);
			}
		
		
	}

	/**
	 * Fill the underlying graph with the values of the underlying dual
	 * variables and, possibly, a fraction (<code>beta</code>) of the edge
	 * factor weights.
	 * 
	 * @param numberOfTokens
	 */
	private void fillGraph(int numberOfTokens) {
		for (int idxHead = 0; idxHead < numberOfTokens; ++idxHead) {
			for (int idxModifier = 0; idxModifier < numberOfTokens; ++idxModifier) {
				double edgeFactorWeight = edgeFactorWeights[idxHead][idxModifier];
				if (Double.isNaN(edgeFactorWeight)) {
					// Skip invalid edges.
					graph[idxHead][idxModifier] = Double.NaN;
					continue;
				}
				// Dual variables.
				graph[idxHead][idxModifier] = dualGrandparentVariables[idxHead][idxModifier]
						+ dualModifierVariables[idxHead][idxModifier];
				// A fraction of the edge factor weight.
				graph[idxHead][idxModifier] += beta * edgeFactorWeight;
			}
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
						grandparentFactorWeightsHeadModifier[idxGrandparent] = convertNan(model
								.getFeatureListScore(ftrs));
						// Loss value for the current edge.
						grandparentFactorWeightsHeadModifier[idxGrandparent] += convertNan(lossWeightEdge);
					} else
						grandparentFactorWeightsHeadModifier[idxGrandparent] = convertNan(Double.NaN);
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
						siblingsFactorWeightsHeadModifier[idxPreviousModifier] = convertNan(model
								.getFeatureListScore(ftrs));
					else
						siblingsFactorWeightsHeadModifier[idxPreviousModifier] = convertNan(Double.NaN);
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
					siblingsFactorWeightsHeadModifier[idxSTART] = convertNan(model
							.getFeatureListScore(ftrs));
				else
					siblingsFactorWeightsHeadModifier[idxSTART] = convertNan(Double.NaN);
			}
		}
		
		// IRVING MUDEI
		siblingsFactorWeights[0][0][0] = 0.0d;
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
	 * perform before returning. That method can still return sooner whenever it
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
	 * Set the value for the beta parameter. This parameter determines the
	 * fraction of the edge factor weights that is passed to the maximum
	 * branching algorithm. The remaining weights are given to the
	 * grandparent/siblings algorithm.
	 * 
	 * @param beta
	 */
	public void setBeta(double beta) {
		this.beta = beta;
		maxGSAlgorithm.setBeta(beta);
	}

	public double getAverageSubGradStepsPerPrediction() {
		return ((double) numSubGradSteps) / ((double) numPredictions);
	}

	/**
	 * Test code.
	 * 
	 * @param args
	 * @throws DPGSException
	 */
	public static void main(String[] args) throws DPGSException {
		// Feature arrays.
		int[][][] eFtrs = new int[5][5][];
		int[][][][] gFtrs = new int[5][5][5][];
		int[][][][] sFtrs = new int[5][6][6][];

		int[] zeroFtrs = new int[] { 0 };

		// Initialize all edge and siblings features.
		for (int idxHead = 0; idxHead < 5; ++idxHead) {
			for (int idxModifier = 0; idxModifier < 6; ++idxModifier) {
				// Edge features.
				eFtrs[idxHead][idxModifier] = zeroFtrs;
				for (int idxPrevModifier = 0; idxPrevModifier < 6; ++idxPrevModifier) {
					sFtrs[idxHead][idxModifier][idxPrevModifier] = zeroFtrs;
				}
			}
		}

		// Initialize all grandparent features.
		for (int idxHead = 0; idxHead < 5; ++idxHead)
			for (int idxModifier = 0; idxModifier < 5; ++idxModifier)
				for (int idxGrandParent = 0; idxGrandParent < 5; ++idxGrandParent)
					gFtrs[idxHead][idxModifier][idxGrandParent] = zeroFtrs;

		// Grandparent ordinary features for 1st tree.
		gFtrs[2][3][0] = new int[] { 10 };
		gFtrs[2][4][0] = new int[] { 10 };

		// Siblings ordinary features for 1st tree.
		sFtrs[0][1][5] = new int[] { 10 };
		sFtrs[0][2][1] = new int[] { 10 };
		sFtrs[0][5][2] = new int[] { 10 };
		sFtrs[2][3][5] = new int[] { 10 };
		sFtrs[2][4][3] = new int[] { 10 };
		sFtrs[2][5][4] = new int[] { 10 };

		// Grandparent ordinary features for 2nd tree.
		gFtrs[2][1][0] = new int[] { 10 };

		// Siblings ordinary features for 2nd tree.
		sFtrs[2][1][2] = new int[] { 20 };
		sFtrs[2][2][1] = new int[] { 0 };

		// Input structure.
		DPGSInput input = new DPGSInput(eFtrs, gFtrs, sFtrs);

		// Inference object.
		DPGSDualInference inference = new DPGSDualInference(input.size());

		// Model.
		DPGSModel model = new DPGSModel(0);

		// Feature weights.
		model.getParameters().put(0, new AveragedParameter(0d));
		model.getParameters().put(10, new AveragedParameter(10d));
		model.getParameters().put(20, new AveragedParameter(20d));
		model.getParameters().put(30, new AveragedParameter(30d));

		// Output to be filled.
		DPGSOutput output = input.createOutput();

		// Inference.
		inference.setMaxNumberOfSubgradientSteps(1000);
		inference.inference(model, input, output);

		output.getHeads()[1] = 2;
		output.getHeads()[2] = 0;
		output.getHeads()[3] = 2;
		output.getHeads()[4] = 2;
		System.out.println(output);
		LOG.info(String
				.format("Optimum found at step %d after %d dual objective increments. Dual objective: %f. Weight: %f",
						-1, -1, -1d, inference.maxGSAlgorithm
								.calcObjectiveValueOfParse(output.getHeads(),
										output.size(),
										inference.edgeFactorWeights,
										inference.grandparentFactorWeights,
										inference.siblingsFactorWeights, null,
										null)));
	}
}
