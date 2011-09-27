package br.pucrio.inf.learn.structlearning.discriminative.data;

/**
 * Input part of an example (X).
 * 
 * @author eraldo
 * 
 */
public interface ExampleInput {

	/**
	 * Return the textual identifier of this example.
	 * 
	 * @return
	 */
	public String getId();

	/**
	 * Create an output object that is compatible with this input, i.e., it can
	 * be used to store an output to this input.
	 * 
	 * @return
	 */
	public ExampleOutput createOutput();

	/**
	 * Normalize the feature vectors within this structure so that the norm of
	 * each one is equal to the given value.
	 * 
	 * The normalization definition is task dependent. In sequence tagging, for
	 * instance, the feature vectors for each token are independently
	 * normalized, i.e., after normalization, the feature vectors of each token
	 * will have the same norm.
	 * 
	 * @param norm
	 */
	public void normalize(double norm);

	/**
	 * Sort feature values to speedup kernel functions computations.
	 */
	public void sortFeatureValues();

}
