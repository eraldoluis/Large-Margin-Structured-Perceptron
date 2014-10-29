package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for dependency parsing with grandparent and sibling
 * features.
 * 
 * This structure stores three main arrays of variables: parse, grandparent and
 * modifiers. The parse array stores the head token for each (modifier) token
 * within the sentence and, thus, stores a feasible parse, that is a rooted tree
 * (branching). The grandparent array stores the (parent) head token of each
 * head token (grandparent of modifiers). For each head token, a modifier array
 * stores which tokens are its modifiers.
 * 
 * @author eraldo
 * 
 */
public class DPGSOutput implements ExampleOutput {

	/**
	 * Head token of each token in the sentence.
	 */
	private int[] heads;

	/**
	 * Parent of each head token (grandparent of its modifiers).
	 */
	private int[] grandparents;

	/**
	 * For each head token, stores which tokens are its modifiers. This is the
	 * siblings structure.
	 */
	private boolean[][] modifiers;

	/**
	 * For each head token, stores which tokens are its modifiers. This is the
	 * siblings structure.
	 */
	private int[][] previousModifiers;

	/**
	 * Allocate an output structure for sentence with the given number of
	 * tokens.
	 * 
	 * @param numberOfTokens
	 */
	public DPGSOutput(int numberOfTokens) {
		heads = new int[numberOfTokens];
		grandparents = new int[numberOfTokens];
		modifiers = new boolean[numberOfTokens][numberOfTokens];
		previousModifiers = new int[numberOfTokens][numberOfTokens];
	}

	@Override
	public ExampleOutput createNewObject() {
		return new DPGSOutput(heads.length);
	}

	/**
	 * Return the number of tokens in this output sentence.
	 * 
	 * @return
	 */
	public int size() {
		return heads.length;
	}

	/**
	 * Return the head of the given token according to the underlying (feasible)
	 * parse structure.
	 * 
	 * @param idxModifier
	 * @return
	 */
	public int getHead(int idxModifier) {
		return heads[idxModifier];
	}

	/**
	 * Set the head of the given modifier.
	 * 
	 * @param idxModifier
	 * @param idxHead
	 */
	public void setHead(int idxModifier, int idxHead) {
		heads[idxModifier] = idxHead;
	}

	/**
	 * Return the internal array of head tokens.
	 * 
	 * @return
	 */
	public int[] getHeads() {
		return heads;
	}

	/**
	 * Return the <b>parent</b> of the given head token (the grandparent of its
	 * children) according to the (not always feasible) grandparent structure.
	 * 
	 * @param idxHead
	 * @return
	 */
	public int getGrandparent(int idxHead) {
		return grandparents[idxHead];
	}

	/**
	 * Set grandparent token for the given head token.
	 * 
	 * @param idxHead
	 */
	public void setGrandparent(int idxHead, int idxGrandparent) {
		grandparents[idxHead] = idxGrandparent;
	}

	/**
	 * Return the internal array of grandparents.
	 * 
	 * @return
	 */
	public int[] getGrandparents() {
		return grandparents;
	}

	/**
	 * Return whether the given modifier token really modifies the given head
	 * token according to the (not always feasible) siblings structure.
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @return
	 */
	public boolean isModifier(int idxHead, int idxModifier) {
		return modifiers[idxHead][idxModifier];
	}

	/**
	 * Set the modifier flag for the given dependency (idxHead, idxModifier).
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param val
	 */
	public void setModifier(int idxHead, int idxModifier, boolean val) {
		modifiers[idxHead][idxModifier] = val;
	}

	/**
	 * Return the internal array of modifiers.
	 * 
	 * @return
	 */
	public boolean[][] getModifiers() {
		return modifiers;
	}

	/**
	 * Set the modifier flag for the given dependency (idxHead, idxModifier).
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param val
	 */
	public void setPreviousModifier(int idxHead, int idxModifier,
			int idxPreviousModifier) {
		previousModifiers[idxHead][idxModifier - 1] = idxPreviousModifier;
	}

	public boolean isPreviousModifier(int idxHead, int idxModifier,
			int idxPreviousModifier) {
		return previousModifiers[idxHead][idxModifier - 1] == idxPreviousModifier;
	}

	/**
	 * Fill grandparent and siblings structures to reflect the heads structure
	 * (the proper, feasible parser).
	 */
	public void fillGSStructuresFromParse() {
		int numTokens = heads.length;
		for (int idxHead = 0; idxHead < numTokens; ++idxHead) {
			grandparents[idxHead] = heads[idxHead];
			for (int idxModifier = 0; idxModifier < numTokens; ++idxModifier)
				modifiers[idxHead][idxModifier] = (heads[idxModifier] == idxHead);
		}
	}

	/**
	 * Return a string representation of this output structure.
	 */
	public String toString() {
		StringBuffer buff = new StringBuffer();
		int numTkns = size();
		// Heads.
		buff.append("Heads       : ");
		for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier)
			buff.append("(" + heads[idxModifier] + "," + idxModifier + ") ");
		// Grandparents.
		buff.append("\nGrandparents: ");
		for (int idxHead = 0; idxHead < numTkns; ++idxHead)
			buff.append("(" + grandparents[idxHead] + "," + idxHead + ") ");
		// Grandparents.
		buff.append("\nModifiers:\n");
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			buff.append("Head " + idxHead + ": ");
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
				if (modifiers[idxHead][idxModifier])
					buff.append(idxModifier + " ");
			}
			buff.append("\n");
		}
		return buff.toString();
	}

	@Override
	public double getFeatureVectorLengthSquared(ExampleInput input,
			ExampleOutput outputPredicted) {

		DPGSOutput predicted = (DPGSOutput) outputPredicted;
		DPGSInput in = (DPGSInput) input;
		Map<Integer, AtomicInteger> featuresMap = new HashMap<Integer, AtomicInteger>();

		/*
		 * For each head and modifier, check whether the predicted factor does
		 * not correspond to the correct one and, then, update the current model
		 * properly.
		 */
		int numTkns = size();
		int updateValue = 1;

		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			// Correct and predicted grandparent heads.
			int correctGrandparent = getHead(idxHead);
			int predictedGrandparent = predicted.getGrandparent(idxHead);

			if (correctGrandparent != predictedGrandparent) {

				if (predictedGrandparent != -1)
					updateFeatureMap(
							in.getEdgeFeatures(predictedGrandparent, idxHead),
							featuresMap, updateValue);

				if (correctGrandparent != -1)
					updateFeatureMap(
							in.getEdgeFeatures(correctGrandparent, idxHead),
							featuresMap, -updateValue);
			}

			/*
			 * Verifiy grandparent and siblings factors for differences between
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
				boolean isCorrectModifier = (isSpecialToken || (this.
						getHead(idxModifier) == idxHead));
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

					if (isCorrectModifier) {

						/*
						 * Current modifier is correct but the predicted
						 * structure does not set it as a modifier of the
						 * current head (false negative). Thus, increment the
						 * weight of both (grandparent and siblings) correct,
						 * but missed, factors.
						 */
						updateFeatureMap(in.getSiblingsFeatures(idxHead,
								idxModifier, correctPreviousModifier),
								featuresMap, -updateValue);

						if (correctGrandparent != -1)
							updateFeatureMap(in.getGrandparentFeatures(idxHead, idxModifier, correctGrandparent),
									featuresMap, -updateValue);
					} else {

						/*
						 * Current modifier is not correct but the predicted
						 * structure does set it as a modifier of the current
						 * head (false positive). Thus, decrement the weight of
						 * both (grandparent and siblings) incorrectly predicted
						 * factors.
						 */
						updateFeatureMap(in.getSiblingsFeatures(idxHead,
								idxModifier, predictedPreviousModifier),
								featuresMap, updateValue);
						if (predictedGrandparent != -1)
							updateFeatureMap(in.getGrandparentFeatures(idxHead,
									idxModifier, predictedGrandparent),
									featuresMap, updateValue);
					}

				} else {
					/*
					 * The current modifier has been correctly predicted for the
					 * current head. Now, additionally check the previous
					 * modifier and the grandparent factor.
					 */

					if (correctPreviousModifier != predictedPreviousModifier) {

						/*
						 * Modifier is correctly predited but previous modifier
						 * is NOT. Thus, the corresponding correct siblings
						 * factor is missing (false negative) and the predicted
						 * one is incorrectly predicted (false positive).
						 */
						updateFeatureMap(in.getSiblingsFeatures(idxHead,
								idxModifier, correctPreviousModifier),
								featuresMap, -updateValue);
						updateFeatureMap(in.getSiblingsFeatures(idxHead,
								idxModifier, predictedPreviousModifier),
								featuresMap, updateValue);
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
							updateFeatureMap(in.getGrandparentFeatures(idxHead, idxModifier, correctGrandparent),
									featuresMap, -updateValue);
						if (predictedGrandparent != -1)
							updateFeatureMap(in.getGrandparentFeatures(idxHead,
									idxModifier, predictedGrandparent),
									featuresMap, updateValue);
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
		
		Collection<AtomicInteger> values = featuresMap.values();
		double difVectorLength = 0;
		
		
		for (AtomicInteger v : values) {
			difVectorLength += Math.pow(v.doubleValue(), 2);
		}

		return difVectorLength;
	}

	private void updateFeatureMap(int[] ftrs,
			Map<Integer, AtomicInteger> featureMap, int updateValue) {
		if (ftrs == null)
			// Inexistent factor. Do nothing.
			return;

		for (int idxFtr = 0; idxFtr < ftrs.length; ++idxFtr) {
			AtomicInteger v = featureMap.get(idxFtr);
			
			if(v == null){
				v = new AtomicInteger(0);
				featureMap.put(idxFtr, v);
			}
			
			v.addAndGet(updateValue);
		}
	}
}
