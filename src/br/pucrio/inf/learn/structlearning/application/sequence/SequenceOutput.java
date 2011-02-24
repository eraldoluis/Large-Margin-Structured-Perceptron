package br.pucrio.inf.learn.structlearning.application.sequence;

import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

/**
 * Sequence of labels for a specific input sequence (tokens).
 * 
 * @author eraldo
 * 
 */
public interface SequenceOutput extends ExampleOutput {

	public int size();

	public int getLabel(int token);

	public void setLabel(int token, int label);

}
