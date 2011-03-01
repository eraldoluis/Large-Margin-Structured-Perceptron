package br.pucrio.inf.learn.structlearning.data;

/**
 * Input part of an example (X).
 * 
 * @author eraldo
 * 
 */
public interface ExampleInput {

	/**
	 * Create an output object that is compatible with this input, i.e., it can
	 * be used to store an output to this input.
	 * 
	 * @return
	 */
	public ExampleOutput createOutput();

}
