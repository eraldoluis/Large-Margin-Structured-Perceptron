package br.pucrio.inf.learn.structlearning.data;

/**
 * Ouput part of an example (Y).
 * 
 * @author eraldo
 * 
 */
public interface ExampleOutput {

	/**
	 * Create a new object of the same type. Usually, this is used to store the
	 * predited values during learning.
	 * 
	 * @return
	 */
	ExampleOutput createNewObject();

	/**
	 * Create a new object with the same content.
	 * 
	 * @return
	 * @throws CloneNotSupportedException
	 */
	Object clone() throws CloneNotSupportedException;

}
