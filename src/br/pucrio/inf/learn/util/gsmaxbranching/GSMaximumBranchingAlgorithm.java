package br.pucrio.inf.learn.util.gsmaxbranching;

import java.util.Arrays;

/**
 * Implement a maximum "branching" algorithm that considers grandparent and
 * siblings factors, as proposed in Koo et al. (EMNLP-2010).
 * 
 * In fact, this algorithm does not guarantee that the final solution is a
 * proper branching (rooted tree). The solution comprises, for each node, a
 * parent node and a list of children, with no guarantee about agreement among
 * nodes. There can be two problems: (i) for some node I and its parent J, it is
 * not guaranteed that I is included in J's children; and (ii) it can exist
 * cycles. This, this algorithm must be used within a dual decomposition
 * algorithm to respect the whole contraints of rooted trees.
 * 
 * @author eraldo
 * 
 */
public class GSMaximumBranchingAlgorithm {

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
	 * Create a new algorithm object whose internal data structures can handle
	 * instances with up to <code>maxNumberOfNodes</code> nodes (or tokens).
	 * This limit can be reassigned later through the method
	 * <code>realloc(maxNumberOfNodes)</code>.
	 * 
	 * @param maxNumberOfNodes
	 */
	public GSMaximumBranchingAlgorithm(int maxNumberOfNodes) {
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
	 * Find the maximum scoring "branching" considering the weights of given
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
	 * @param grandparentWeights
	 *            weights of the grandparent factors. The paramters of these
	 *            factors are
	 *            <code>(idxHead, idxModifier, idxGrandparent)</code>.
	 * @param siblingsWeights
	 *            weights of the siblings factors. The parameters of these
	 *            factors are
	 *            <code>(idxHead, idxModifier, idxPreviousModifier)</code>.
	 * @param grandparents
	 *            output array that will be filled with the grandparent for each
	 *            head node.
	 * @param modifiers
	 *            output array that will be filled with the modifiers for each
	 *            head node.
	 * @return the weight of the best scoring solution, that is the sum of the
	 *         weights of all factors included in this solution.
	 */
	public double findMaximumBranching(int numberOfNodes,
			double[][][] grandparentWeights, double[][][] siblingsWeights,
			int[] grandparents, boolean[][] modifiers) {
		double weight = 0d;
		for (int idxHead = 0; idxHead < numberOfNodes; ++idxHead)
			weight += findMaximumBranchingForHead(numberOfNodes, idxHead,
					grandparentWeights[idxHead], siblingsWeights[idxHead],
					grandparents, modifiers);
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
	 * @param grandparentWeights
	 *            weights of the grandparent factors for the given head node.
	 *            The paramters of these factors are
	 *            <code>(idxModifier, idxGrandparent)</code>.
	 * @param siblingsWeights
	 *            weights of the siblings factors for the given head node. The
	 *            parameters of these factors are
	 *            <code>(idxHead, idxModifier, idxPreviousModifier)</code>.
	 * @param grandparents
	 *            output array that will be filled with the grandparent for each
	 *            head node.
	 * @param modifiers
	 *            output array that will be filled with the modifiers for each
	 *            head node.
	 * @return the weight of the included factors for the given head node.
	 */
	public double findMaximumBranchingForHead(int numberOfNodes, int idxHead,
			double[][] grandparentWeights, double[][] siblingsWeights,
			int[] grandparents, boolean[][] modifiers) {
		double bestWeight = Double.NEGATIVE_INFINITY;
		int bestGrandParent = -1;
		for (int idxGradparent = 0; idxGradparent < numberOfNodes; ++idxGradparent) {
			// Skip infeasile self-loop arcs.
			if (idxGradparent == idxHead)
				continue;

			double weight = 0d;

			// Process the modifiers on the LEFT side of the current head token.
			for (int idxModifier = 0; idxModifier < idxHead; ++idxModifier) {
				double wGrandparentFactor = grandparentWeights[idxModifier][idxGradparent];
				if (Double.isNaN(wGrandparentFactor)) {
					// Grandparent is not valid, thus skip this modifier.
					previousModifiers[idxModifier] = -1;
					accumWeights[idxModifier] = Double.NaN;
					continue;
				}

				/*
				 * Find best previous modifiers for the current modifier (that
				 * lies on the LEFT side of the current head). It fills the
				 * previousModifiers and accumWeights arrays with the found
				 * values.
				 */
				findBestPreviousModifier(0, idxHead, idxModifier,
						siblingsWeights[idxModifier]);

				/*
				 * Gradparent factor is fixed across siblings modifiers for the
				 * current modifier. Thus, we can include the corresponding
				 * (grandparent) weight here, after calculating the best
				 * sequence of sibling modifiers.
				 */
				if (!Double.isNaN(accumWeights[numberOfNodes]))
					accumWeights[numberOfNodes] += wGrandparentFactor;
			}

			/*
			 * Find the complete, best sequence of modifiers on the LEFT side of
			 * the current head. We represent the last factor (for LEFT
			 * modifiers) by the special index 'idxHead' that is used for the
			 * START and END special symbols.
			 */
			findBestPreviousModifier(0, idxHead, idxHead,
					siblingsWeights[idxHead]);

			// Process the modifiers on the RIGHT side of the current head.
			for (int idxModifier = idxHead + 1; idxModifier < numberOfNodes; ++idxModifier) {
				double wGrandparentFactor = grandparentWeights[idxModifier][idxGradparent];
				if (Double.isNaN(wGrandparentFactor)) {
					// Grandparent is not valid, thus skip this modifier.
					previousModifiers[idxModifier] = -1;
					accumWeights[idxModifier] = Double.NaN;
					continue;
				}

				/*
				 * Find best previous modifiers for the current modifier (that
				 * lies on the RIGHT side of the current head). It fills the
				 * previousModifiers and accumWeights arrays with the found
				 * values.
				 */
				findBestPreviousModifier(idxHead + 1, numberOfNodes,
						idxModifier, siblingsWeights[idxModifier]);

				/*
				 * Gradparent factor is fixed across siblings modifiers for the
				 * current modifier. Thus, we can include the corresponding
				 * (grandparent) weight here, after calculating the best
				 * sequence of sibling modifiers.
				 */
				if (!Double.isNaN(accumWeights[numberOfNodes]))
					accumWeights[numberOfNodes] += wGrandparentFactor;
			}

			/*
			 * Find the complete, best sequence of modifiers on the RIGHT side
			 * of the current head. We represent the last factor (for RIGHT
			 * modifiers) by the special index 'numberOfTokens' that is used for
			 * the START and END special symbols.
			 */
			findBestPreviousModifier(0, idxHead, idxHead,
					siblingsWeights[idxHead]);

			// Store the best solution among all grandparents.
			if (weight > bestWeight) {
				for (int idx = 0; idx <= numberOfNodes; ++idx)
					bestPreviousModifiers[idx] = previousModifiers[idx];
				bestWeight = weight;
				bestGrandParent = idxGradparent;
			}
		}

		/*
		 * Fill the output variables for the current head (grandparents and
		 * modifiers arrays) with the best solution among all grandparents.
		 */
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
			// Current siblings factor weight.
			double wSiblingsFactor = siblingsWeights[idxPrevModifier];

			if (Double.isNaN(wSiblingsFactor))
				// Skip inexistent factors.
				continue;

			double accumWeight = accumWeights[idxPrevModifier];
			if (Double.isNaN(accumWeight))
				// Skip inexistent path.
				continue;

			accumWeight += wSiblingsFactor;

			if (Double.isNaN(bestAccumWeight) || accumWeight > bestAccumWeight) {
				// Found a better path.
				bestAccumWeight = accumWeight;
				bestPreviousModifier = idxPrevModifier;
			}
		}

		if (Double.isNaN(bestAccumWeight)) {
			/*
			 * No factor is valid for the current pair given by (idxHead,
			 * idxModifier, *).
			 */
			previousModifiers[idxModifier] = -1;
			accumWeights[idxModifier] = Double.NaN;
		} else {
			/*
			 * Store the best previous modifier for the current pair given by
			 * (idxHead, idxModifier).
			 */
			previousModifiers[idxModifier] = bestPreviousModifier;
			accumWeights[idxModifier] = bestAccumWeight;
		}
	}
}
