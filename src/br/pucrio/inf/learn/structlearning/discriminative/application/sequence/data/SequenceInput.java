package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Input sequence structure.
 * 
 * It is composed by a sequence of tokens. Each token comprises a sparse feature
 * vector. This sparse vector is represented by a vector of features (codes) and
 * another vector with the corresponding weight for each feature. Only features
 * whose weight is different of zero are represented. All absent features are
 * assumed to have zero weights.
 * 
 * @author eraldo
 * 
 */
public interface SequenceInput extends ExampleInput {

	/**
	 * Return the number of tokens in this sequence.
	 * 
	 * @return
	 */
	public int size();

	/**
	 * Return the number of input features (discounting the possible label
	 * information) in the given token.
	 * 
	 * @param token
	 * @return
	 */
	public int getNumberOfInputFeatures(int token);

	/**
	 * Return the feature in the given index of the given token.
	 * 
	 * @param token
	 * @param index
	 * @return
	 */
	public int getFeature(int token, int index);

	/**
	 * Return the weight associated with the feature in the given index.
	 * 
	 * @param token
	 * @param index
	 * @return
	 */
	public double getFeatureWeight(int token, int index);

	/**
	 * Return an iterator for the features of the given token.
	 * 
	 * @param token
	 * @return
	 */
	public Iterable<Integer> getFeatureCodes(int token);

}
