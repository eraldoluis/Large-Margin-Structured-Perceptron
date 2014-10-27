package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

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
	public void setPreviousModifier(int idxHead, int idxModifier, int idxPreviousModifier) {
		previousModifiers[idxHead][idxModifier - 1] = idxPreviousModifier;
	}

	public boolean isPreviousModifier(int idxHead,int idxModifier, int idxPreviousModifier) {
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
}
