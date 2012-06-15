package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for dependency parsing with grandparent and sibling
 * features.
 * 
 * This structure stores three main arrays of variables: complete and feasible
 * parse, grandparent and modifiers. The parse array stores the head token for
 * each (modifier) token within the sentence. The grandparent array stores the
 * (parent) head token of each head token (grandparent of sinblings). For each
 * head token, a sibling array stores which tokens are its modifiers.
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
	 * Parent of each head token.
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
}
