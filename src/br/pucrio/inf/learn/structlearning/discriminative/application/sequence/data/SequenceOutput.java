package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Sequence of labels for an input sequence.
 * 
 * @author eraldo
 * 
 */
public interface SequenceOutput extends ExampleOutput {

	/**
	 * Return the number of tokens in this sequence.
	 * 
	 * @return
	 */
	public int size();

	/**
	 * Return the label of the given token.
	 * 
	 * @param token
	 * @return
	 */
	public int getLabel(int token);

	/**
	 * Set the label for the given token.
	 * 
	 * @param token
	 * @param label
	 */
	public void setLabel(int token, int label);

}
