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
	 * Allocate an output structure for sentence with the given number of
	 * tokens.
	 * 
	 * @param numberOfTokens
	 */
	public DPGSOutput(int numberOfTokens) {
		heads = new int[numberOfTokens];
		grandparents = new int[numberOfTokens];
		modifiers = new boolean[numberOfTokens][numberOfTokens];
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
	 * Return the internal array of head tokens.
	 * 
	 * @return
	 */
	public int[] getHeads() {
		return heads;
	}

	/**
	 * Return the parent of the given head token (and, consequently, grandparent
	 * of its children) according to the (not always feasible) grandparent
	 * structure.
	 * 
	 * @param idxHead
	 * @return
	 */
	public int getGrandparent(int idxHead) {
		return grandparents[idxHead];
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
	 * Return the internal array of modifiers.
	 * 
	 * @return
	 */
	public boolean[][] getModifiers() {
		return modifiers;
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
		buff.append("Heads: ");
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
