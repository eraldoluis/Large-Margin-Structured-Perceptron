package br.pucrio.inf.learn.structlearning.discriminative.application.dp.data;

import java.io.Serializable;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Represent a branching of a complete graph of a sentence. Since each node in a
 * branching can have, at most, one parent node, we represent the inversed
 * branching as an array of parents (heads).
 * 
 * @author eraldo
 * 
 */
public class DPOutput implements ExampleOutput, Serializable {

	/**
	 * Automatically generated serial version id.
	 */
	private static final long serialVersionUID = -8372776458147079713L;

	/**
	 * Indicate the head (parent) of each token in a sentence.
	 */
	private int[] heads;

	/**
	 * Create an object to represent a branching on a graph with the given
	 * number of nodes.
	 * 
	 * @param numberOfTokens
	 */
	public DPOutput(int numberOfTokens) {
		heads = new int[numberOfTokens];
	}

	@Override
	public DPOutput createNewObject() {
		return new DPOutput(heads.length);
	}

	/**
	 * Return the inverted branching that is represented by an array of head
	 * tokens.
	 * 
	 * @return
	 */
	public int[] getInvertedBranchingArray() {
		return heads;
	}

	/**
	 * @param token
	 * @return the head of the given token
	 */
	public int getHead(int token) {
		return heads[token];
	}

	/**
	 * Set the head of a token.
	 * 
	 * @param token
	 * @param head
	 */
	public void setHead(int token, int head) {
		heads[token] = head;
	}

}
