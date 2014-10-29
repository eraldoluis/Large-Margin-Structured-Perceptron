package br.pucrio.inf.learn.structlearning.discriminative.data;


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
	
	
	double getFeatureVectorLengthSquared(ExampleInput input, ExampleOutput other);
}
