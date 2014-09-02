package br.pucrio.inf.learn.util.gsmaxbranching;

import java.util.Arrays;

/**
 * Implement a maximum grandparent and siblings algorithm as proposed in Koo et
 * al. (EMNLP-2010).
 * 
 * The solution comprises, for each head node, a parent node and a sequence of
 * left and right children (modifiers) nodes with maximum weight according to
 * grandparent and siblings factors. Grandparent factors have parameters
 * (idxHead, idxModifier, idxGrandparent), where idxHead is the index of a head
 * token, idxModifier is the index of a modifier token and idxGrandparent is the
 * index of the head token of the head token, that is the grandparent token of
 * the modifier. Siblings factors have parameters (idxHead, idxModifier,
 * idxPreviousModifier), where, idxPreviousModfier is the index a the closest
 * modifier of idxModifier related to the corresponding head.
 * 
 * @author eraldo
 * 
 */
public class MaximumGrandparentSiblingsAlgorithm {

	/**
	 * Dynamic programming table that stores the back pointers with the best
	 * previous modifier for each modifier of the current considered head.
	 * 
	 * For each head, this array and the siblings factor weight arrays
	 * (siblingsWeights) are split in two slices exactly in the current head
	 * index (idxHead). The idxHead index does not contain any valid node, since
	 * it is impossible to connect a node to itself (there is no self-loop arc).
	 * On the other hand, the siblings model considers two special nodes, START
	 * and END, that represent the fixed, and artificial, first and last nodes
	 * of any sequence of modifiers. Thus, for left modifiers, we use the
	 * idxHead index to represent the START and END symbols for left modifiers
	 * (modifiers that lie on the left side of the current head). Additionally,
	 * we use the numberOfNodes index to represent the START and END symbols for
	 * RIGHT modifiers. Therefore, the arrays sinblingsWeights and
	 * previousModifiers must contain (numberOfNodes + 1) positions.
	 */
	private int[] previousModifiers;

	/**
	 * Dynamic programming table that stores the accumulated weight up to each
	 * modifier for the best solution that includes the corresponding modifier.
	 * The indexes idxHead and numberOfNodes are used to store the weight of the
	 * two final solutions, for left and right modifiers, respectively.
	 */
	private double[] accumWeights;

	/**
	 * Store the best solution of modifiers (in the same format as in
	 * <code>previousModifiers</code>) among all grandparents.
	 */
	private int[] bestPreviousModifiers;

	/**
	 * Fraction of the edge factor weights to be used in the maximum branching
	 * algorithm. The remaining weights (<code>1-beta</code>) is used in this
	 * algorithm.
	 */
	private double beta;

	/**
	 * Create a new algorithm object whose internal data structures can handle
	 * instances with up to <code>maxNumberOfNodes</code> nodes (or tokens).
	 * This limit can be reassigned later through the method
	 * <code>realloc(maxNumberOfNodes)</code>.
	 * 
	 * @param maxNumberOfNodes
	 */
	public MaximumGrandparentSiblingsAlgorithm(int maxNumberOfNodes) {
		this.beta = 0d;
		realloc(maxNumberOfNodes);
	}

	/**
	 * Realloc memory for the internal data structures to handle instances with
	 * up to <code>maxNumberOfNodes</code> nodes (or tokens).
	 * 
	 * @param maxNumberOfNodes
	 *            maximum number of nodes supported by the internal data
	 *            structures.
	 */
	public void realloc(int maxNumberOfNodes) {
		accumWeights = new double[maxNumberOfNodes + 1];
		previousModifiers = new int[maxNumberOfNodes + 1];
		bestPreviousModifiers = new int[maxNumberOfNodes + 1];
	}

	/**
	 * Find the maximum scoring structure considering the weights of given
	 * factors: <code>grandparentWeights</code> (with parameters
	 * <code>(idxHead, idxModifier, idxGrandparent)</code> and
	 * <code>siblingsWeights</code> (with parameters
	 * <code>(idxHead, idxModifier, idxPreviousModifier)</code>.
	 * 
	 * The best scoring solution is filled in the arrays
	 * <code>grandparents</code> and <code>modifiers</code>. The former stores
	 * the chosen grandparents for each head node by just seting the chose
	 * index. The later stores the chosen modifiers for each head node by seting
	 * to <code>true</code> the indexes of the corresponding modifiers.
	 * 
	 * @param numberOfNodes
	 *            number of nodes in the given instance.
	 * @param edgeFactorWeights
	 *            weights of the edge factors. The parameters of these factors
	 *            are (idxHead, idxModifier).
	 * @param grandparentFactorWeights
	 *            weights of the grandparent factors. The paramters of these
	 *            factors are (idxHead, idxModifier, idxGrandparent).
	 * @param siblingsFactorWeights
	 *            weights of the siblings factors. The parameters of these
	 *            factors are (idxHead, idxModifier, idxPreviousModifier).
	 * @param dualGrandparentVars
	 *            dual variable for each grandparent constraint identified by
	 *            (idxHead, idxGrandparent).
	 * @param dualModifierVars
	 *            dual variable for each modifier constraint identified by
	 *            (idxHead, idxModifier).
	 * @param grandparents
	 *            output array that will be filled with the grandparent for each
	 *            head node.
	 * @param modifiers
	 *            output array that will be filled with the modifiers for each
	 *            head node.
	 * @return the weight of the best scoring solution, that is the sum of the
	 *         weights of all factors included in this solution.
	 */
	public double findMaximumGrandparentSiblings(int numberOfNodes,
			double[][] edgeFactorWeights,
			double[][][] grandparentFactorWeights,
			double[][][] siblingsFactorWeights, double[][] dualGrandparentVars,
			double[][] dualModifierVars, int[] grandparents,
			boolean[][] modifiers) {
		double weight = 0d;
		
		for (int idxHead = 0; idxHead < numberOfNodes; ++idxHead){
			weight += findMaximumGrandparentSiblingsForHead(numberOfNodes,
					idxHead, edgeFactorWeights,
					grandparentFactorWeights[idxHead],
					siblingsFactorWeights[idxHead], dualGrandparentVars,
					dualModifierVars, grandparents, modifiers);
		}
		
		return weight;
	}

	/**
	 * Find the maximum scoring grandparent and sequence of modifiers for the
	 * head node <code>idxHead</code>.
	 * 
	 * The best scoring solution is filled in the arrays
	 * <code>grandparents</code> and <code>modifiers</code>. The former stores
	 * the chosen grandparent index in the index of the given head node. The
	 * later stores the chosen modifiers by seting to <code>true</code> the
	 * indexes of the corresponding modifiers.
	 * 
	 * @param numberOfNodes
	 *            number of nodes in the given instance.
	 * @param idxHead
	 *            index of the head node to be considered.
	 * @param edgeFactorWeightsForHead
	 *            weights of the edge factors for the given head node. The
	 *            parameters of these factors, given a fixed head, is only
	 *            (idxModifier).
	 * @param grandparentFactorWeightsForHead
	 *            weights of the grandparent factors for the given head node.
	 *            The paramters of these factors, given a fixed head, are
	 *            (idxModifier, idxGrandparent).
	 * @param siblingsFactorWeightsForHead
	 *            weights of the siblings factors for the given head node. The
	 *            parameters of these factors, given a fixed head, are
	 *            (idxModifier, idxPreviousModifier).
	 * @param dualGrandparentVars
	 *            dual variables for the grandparent constraints.
	 * @param dualModifierVars
	 *            dual variables for the modifier constraints.
	 * @param grandparents
	 *            output array that will be filled with the grandparent for each
	 *            head node.
	 * @param modifiers
	 *            output array that will be filled with the modifiers for each
	 *            head node.
	 * @return the weight of the included factors for the given head node.
	 */
	public double findMaximumGrandparentSiblingsForHead(int numberOfNodes,
			int idxHead, double[][] edgeFactorWeights,
			double[][] grandparentFactorWeightsForHead,
			double[][] siblingsFactorWeightsForHead,
			double[][] dualGrandparentVars, double[][] dualModifierVars,
			int[] grandparents, boolean[][] modifiers) {
		/*
		 * Initialize the best left and right modifier sequence to an empty.
		 * Just in case, if no sequence (or even no grandparent) is feasible
		 * (when all factors have weights equal to Double.NaN).
		 */
		bestPreviousModifiers[idxHead] = idxHead;
		bestPreviousModifiers[numberOfNodes] = numberOfNodes;

		double bestWeight = Double.NEGATIVE_INFINITY;
		int bestGrandParent = -1;
		for (int idxGrandparent = -1; idxGrandparent < numberOfNodes; ++idxGrandparent) {
			/*
			 * Weight of the complete solution for (i) the given head, (ii) the
			 * current grandparent, and (iii) the maximum scoring sequence of
			 * modifiers. It starts from the weight of the grandparent variable
			 * because this weight depends only on (idxHead, idxGrandparent).
			 */
			double weight = 0d;
			if (idxGrandparent != -1) {
				if (dualGrandparentVars != null)
					weight = -dualGrandparentVars[idxGrandparent][idxHead];
//
//				/*
//				 * TODO test. Additionally, we include the edge factor using the
//				 * grandparent instead of using the modifier. This is done to
//				 * avoid the issue of missing a grandparent factors on leaf
//				 * tokens. In these cases, it is impossible to select the right
//				 * grandparent (it will always be -1).
//				 */
				double edgeFactorWeight = (1 - beta)
						* edgeFactorWeights[idxGrandparent][idxHead];
				if (Double.isNaN(edgeFactorWeight)) {
					/*
					 * Edge factor is invalid. Thus, the current grandparent is
					 * invalid.
					 */
					continue;
					// // TODO test
					// edgeFactorWeight = 0d;
				}

				weight += edgeFactorWeight;
			}

			// Process the modifiers on the LEFT side of the current head token.
			for (int idxModifier = 0; idxModifier < idxHead; ++idxModifier) {
				/*
				 * Grandparent factor weight, which is fixed for the current
				 * modifier. That is it does not consider the siblings of this
				 * modifier. However, if the weight for this factor is NaN, then
				 * the current modifier cannot be included and, thus, no sibling
				 * of its needs to be considered.
				 */
				double wGrandparentFactor = 0d;
				if (idxGrandparent != -1)
					wGrandparentFactor = grandparentFactorWeightsForHead[idxModifier][idxGrandparent];
				if (Double.isNaN(wGrandparentFactor)) {
					// Grandparent is not valid, thus skip this modifier.
					previousModifiers[idxModifier] = -1;
					accumWeights[idxModifier] = Double.NaN;
					continue;
					// // TODO test
					// wGrandparentFactor = 0d;
				}

				/*
				 * Find best previous modifiers for the current modifier (that
				 * lies on the LEFT side of the current head). It fills the
				 * previousModifiers and accumWeights arrays with the found
				 * values.
				 */
				findBestPreviousModifier(0, idxHead, idxModifier,
						siblingsFactorWeightsForHead[idxModifier]);

				/*
				 * Gradparent factor is fixed across siblings modifiers for the
				 * current modifier. Thus, we can include the corresponding
				 * (grandparent) weight here, after calculating the best
				 * sequence of sibling modifiers.
				 */
				accumWeights[idxModifier] += wGrandparentFactor;

//				/*
//				* Same thing for the edge factor weight that depends only on
//				* (idxHead, idxModifier).
//				*/
//				
//				double edgeFactorWeight = (1 - beta)
//				 * edgeFactorWeights[idxHead][idxModifier];
//				if (Double.isNaN(edgeFactorWeight)) {
//					// // Edge factor is invalid.
//					previousModifiers[idxModifier] = -1;
//					accumWeights[idxModifier] = Double.NaN;
//					continue;
//					// // // TODO test
//					//edgeFactorWeight = 0d;
//				}
//				
//				accumWeights[idxModifier] += edgeFactorWeight;

				/*
				 * Same thing for the modifier dual var that depends only on
				 * (idxHead, idxModifier).
				 */
				if (dualModifierVars != null)
					accumWeights[idxModifier] -= dualModifierVars[idxHead][idxModifier];
			}

			/*
			 * Find the complete, best sequence of modifiers on the LEFT side of
			 * the current head. We represent the last factor (for LEFT
			 * modifiers) by the special index 'idxHead' that is used for the
			 * START and END special symbols.
			 */
			findBestPreviousModifier(0, idxHead, idxHead,
					siblingsFactorWeightsForHead[idxHead]);

			// Add the weight of the best left sequence of modifiers.
			weight += accumWeights[idxHead];

			// Process the modifiers on the RIGHT side of the current head.
			for (int idxModifier = idxHead + 1; idxModifier < numberOfNodes; ++idxModifier) {
				/*
				 * Grandparent factor weight, which is fixed for the current
				 * modifier. That is it does not consider the siblings of this
				 * modifier. However, if the weight for this factor is NaN, then
				 * the current modifier cannot be included and, thus, no sibling
				 * of its needs to be considered.
				 */
				double wGrandparentFactor = 0d;
				if (idxGrandparent != -1)
					wGrandparentFactor = grandparentFactorWeightsForHead[idxModifier][idxGrandparent];
				if (Double.isNaN(wGrandparentFactor)) {
					// Grandparent is not valid, thus skip this modifier.
					previousModifiers[idxModifier] = -1;
					accumWeights[idxModifier] = Double.NaN;
					continue;
					// // TODO test
					// wGrandparentFactor = 0d;
				}

				/*
				 * Find best previous modifiers for the current modifier (that
				 * lies on the RIGHT side of the current head). It fills the
				 * previousModifiers and accumWeights arrays with the found
				 * values.
				 */
				findBestPreviousModifier(idxHead + 1, numberOfNodes,
						idxModifier, siblingsFactorWeightsForHead[idxModifier]);

				/*
				 * Gradparent factor is fixed across siblings modifiers for the
				 * current modifier. Thus, we can include the corresponding
				 * (grandparent) weight here, after calculating the best
				 * sequence of sibling modifiers.
				 */
				accumWeights[idxModifier] += wGrandparentFactor;

				
//				 /* Same thing for the edge factor weight that depends only on
//				 * (idxHead, idxModifier).
//				 */
//				 double edgeFactorWeight = (1 - beta)
//						 * edgeFactorWeights[idxHead][idxModifier];
//				 
//				 if (Double.isNaN(edgeFactorWeight)) {
//					 //Edge factor is invalid.
//					 previousModifiers[idxModifier] = -1;
//					 accumWeights[idxModifier] = Double.NaN;
//					 continue;
//					 //TODO test
//					 edgeFactorWeight = 0d;
//				 }
//				
//				 accumWeights[idxModifier] += edgeFactorWeight;
				 
				/*
				 * Same thing for the modifier dual var that depends only on
				 * (idxHead, idxModifier).
				 */
				if (dualModifierVars != null)
					accumWeights[idxModifier] -= dualModifierVars[idxHead][idxModifier];
			}

			/*
			 * Find the complete, best sequence of modifiers on the RIGHT side
			 * of the current head. We represent the last factor (for RIGHT
			 * modifiers) by the special index 'numberOfTokens' that is used for
			 * the START and END special symbols.
			 */
			findBestPreviousModifier(idxHead + 1, numberOfNodes, numberOfNodes,
					siblingsFactorWeightsForHead[numberOfNodes]);

			// Add weight of the best sequence of right modifiers.
			weight += accumWeights[numberOfNodes];

			// Store the best solution among all grandparents.
			if (weight > bestWeight) {
				for (int idx = 0; idx <= numberOfNodes; ++idx)
					bestPreviousModifiers[idx] = previousModifiers[idx];
				bestWeight = weight;
				bestGrandParent = idxGrandparent;
			}
		}

		/*
		 * Fill the output variables for the current head (grandparents and
		 * modifiers arrays) with the best solution among all grandparents.
		 */
		if (bestGrandParent == idxHead)
			bestGrandParent = -1;
		grandparents[idxHead] = bestGrandParent;
		// Clear modifiers array.
		Arrays.fill(modifiers[idxHead], false);
		// Fill modifiers array with LEFT modifiers.
		int idxModifier = bestPreviousModifiers[idxHead];
		while (idxModifier != idxHead) {
			modifiers[idxHead][idxModifier] = true;
			idxModifier = bestPreviousModifiers[idxModifier];
		}
		// Fill modifiers array with RIGHT modifiers.
		idxModifier = bestPreviousModifiers[numberOfNodes];
		while (idxModifier != numberOfNodes) {
			modifiers[idxHead][idxModifier] = true;
			idxModifier = bestPreviousModifiers[idxModifier];
		}
		
		// IRVING MUDEI
		if(bestWeight == Double.NEGATIVE_INFINITY){
			bestWeight= 0.0d;
		}

		// Return the weight of the best sequence of left and right modifiers.
		return bestWeight;
	}

	/**
	 * Find the best scoring previous modifier for the modifier in the index
	 * <code>idxModifier</code>. These modifiers can lie on the left or on the
	 * right side of the head node.
	 * 
	 * @param firstIndex
	 *            indicates the first index to be considered within the factors
	 *            weights given in <code>siblingsWeights</code>. This parameter
	 *            is necessary because this method is used for left and right
	 *            modifiers.
	 * @param startEndIndex
	 *            indicates the START and END special symbols index within the
	 *            weights factor array. This is also the limit of the feasible
	 *            modifiers. This index can be either: the index of the current
	 *            head (for left modifiers) or the number of tokens for the
	 *            current instance (for right modifiers).
	 * @param idxModifier
	 *            index of the modifier to be considered.
	 * @param siblingsWeights
	 *            weights of the sinblings factors for the current head and the
	 *            given modifier (<code>idxModifier</code>).
	 */
	protected void findBestPreviousModifier(int firstIndex, int startEndIndex,
			int idxModifier, double[] siblingsWeights) {
		// Start with the START symbol as the best previous modifier.
		int bestPreviousModifier = startEndIndex;
		double bestAccumWeight = siblingsWeights[startEndIndex];

		// Find the best previous modifier.
		for (int idxPrevModifier = firstIndex; idxPrevModifier < idxModifier; ++idxPrevModifier) {
			/*
			 * Weight of the siblings factor associated with the previous
			 * modifier index.
			 */
			double wSiblingsFactor = siblingsWeights[idxPrevModifier];
			if (Double.isNaN(wSiblingsFactor))
				// Skip inexistent factors.
				continue;
			// // TODO test
			// wSiblingsFactor = 0d;

			// Weight accumulated up to the previous modifier.
			double accumWeight = accumWeights[idxPrevModifier];
			if (Double.isNaN(accumWeight))
				// Skip inexistent path.
				continue;
			// // TODO test
			// accumWeight = 0d;

			// Accumulated weight up to the modifier.
			accumWeight += wSiblingsFactor;

			if (Double.isNaN(bestAccumWeight) || accumWeight > bestAccumWeight) {
				// Found a better path.
				bestAccumWeight = accumWeight;
				bestPreviousModifier = idxPrevModifier;
			}
		}

		/*
		 * Store the best previous modifier for the current pair given by
		 * (idxHead, idxModifier).
		 */
		previousModifiers[idxModifier] = bestPreviousModifier;
		// if (Double.isNaN(bestAccumWeight))
		// bestAccumWeight = 0d;
		accumWeights[idxModifier] = bestAccumWeight;
	}

	/**
	 * Calculate the weight of the given parse tree <code>heads</code> according
	 * to the given factor weights and dual variables. That is the
	 * grandparent/siblings objective function minus dual variables. If the user
	 * wants the GS objective function value alone, then it can give
	 * <code>null</code> values for the dual variable arrays (
	 * <code>dualGrandparentVars</code> and <code>dualModifierVars</code>).
	 * 
	 * @param heads
	 *            parse tree to measure the objective function value.
	 * @param numberOfNodes
	 *            number of nodes (tokens) in the given parse tree.
	 * @param edgeFactorWeights
	 *            weights of edge factors.
	 * @param grandparentFactorWeights
	 *            weights of grandparent factors.
	 * @param siblingsFactorWeights
	 *            weights of siblings factors.
	 * @param dualGrandparentVars
	 *            dual variable values corresponding to grandparent constraints.
	 *            It can be null if the user wants the value of the GS objective
	 *            function alone.
	 * @param dualModifierVars
	 *            dual variable values corresponding to modifier constraints. It
	 *            can be <code>null</code> if the user wants the value of the GS
	 *            objective function alone.
	 * @return the weight of the given parse according to the given factor
	 *         weights and dual variables.
	 */
	public double calcObjectiveValueOfParse(int heads[], int numberOfNodes,
			double[][] edgeFactorWeights,
			double[][][] grandparentFactorWeights,
			double[][][] siblingsFactorWeights, double[][] dualGrandparentVars,
			double[][] dualModifierVars) {
		// Total weight of the given parse.
		double weight = 0d;
		double w;
		double l;
		double [] wh = new double [numberOfNodes];
		
		l = weight;

		for (int idxHead = 0; idxHead < numberOfNodes; ++idxHead) {
			// Parent of the current head (grandparent of its modifiers).
			int idxGrandparent = heads[idxHead];
			
			if (idxGrandparent != -1 && dualGrandparentVars != null)
				// Grandparent dual variable.
				weight -= dualGrandparentVars[idxGrandparent][idxHead];
			
			if (idxGrandparent != -1) {
				/*
				 * TODO test
				 */
				w = (1 - beta)
						* edgeFactorWeights[idxGrandparent][idxHead];
				if (!Double.isNaN(w)) {
					weight += w;
				}
			}
			
			// LEFT modifiers. The special START symbol is equal to idxHead.
			int idxPrevModifier = idxHead;
			for (int idxModifier = 0; idxModifier < idxHead; ++idxModifier) {
				if (heads[idxModifier] != idxHead)
					// It is not a modifier of the current head.
					continue;

				if (idxGrandparent != -1) {
					// Grandparent factor.
					w = grandparentFactorWeights[idxHead][idxModifier][idxGrandparent];
					if (!Double.isNaN(w))
						weight += w;
				}

				// Edge factor.
//				w = (1 - beta) * edgeFactorWeights[idxHead][idxModifier];
//				if (!Double.isNaN(w))
//					weight += w;

				// Sibling factor.
				w = siblingsFactorWeights[idxHead][idxModifier][idxPrevModifier];
				if (!Double.isNaN(w))
					weight += w;

				// Modifier dual variable.
				if (dualModifierVars != null)
					weight -= dualModifierVars[idxHead][idxModifier];

				// Update previous modifier.
				idxPrevModifier = idxModifier;
			}

			// Special END symbol that is equal is equal to idxHead.
			w = siblingsFactorWeights[idxHead][idxHead][idxPrevModifier];
			if (!Double.isNaN(w))
				weight += w;

			// RIGHT modifiers. The START symbol is equal to numberOfNodes.
			idxPrevModifier = numberOfNodes;
			for (int idxModifier = idxHead + 1; idxModifier < numberOfNodes; ++idxModifier) {
				if (heads[idxModifier] != idxHead)
					// It is not a modifier of the current head.
					continue;

				if (idxGrandparent != -1) {
					// Grandparent factor.
					w = grandparentFactorWeights[idxHead][idxModifier][idxGrandparent];
					if (!Double.isNaN(w))
						weight += w;
				}

				// Edge factor.
//				w = (1 - beta) * edgeFactorWeights[idxHead][idxModifier];
//				if (!Double.isNaN(w))
//					weight += w;

				// Sibling factor.
				w = siblingsFactorWeights[idxHead][idxModifier][idxPrevModifier];
				if (!Double.isNaN(w))
					weight += w;

				// Modifier dual variable.
				if (dualModifierVars != null)
					weight -= dualModifierVars[idxHead][idxModifier];

				// Update previous modifier.
				idxPrevModifier = idxModifier;
			}

			// Special END symbol that is equal is equal to numberOfNodes.
			w = siblingsFactorWeights[idxHead][numberOfNodes][idxPrevModifier];
			if (!Double.isNaN(w))
				weight += w;
			
			wh[idxHead] = weight - l;
			l = weight;
		}

		return weight;
	}

	/**
	 * Set the fraction of edge factor weights to be used in the maximum
	 * branching algorithm. The remaining fraction (<code>1-beta</code>) is used
	 * in this algorithm.
	 * 
	 * @param val
	 */
	public void setBeta(double val) {
		beta = val;
	}
}
