package br.pucrio.inf.learn.structlearning.generative.data;

/**
 * Represents an example within a dataset.
 * 
 * Usually, an example is a sentence but can be a paragraph or even a whole
 * document. An example comprises a sequence of tokens and each token is a set
 * of feature along their values. This is an encoded representation, i.e., all
 * feature values are represented as integers, altough the user can request the
 * string representation of some value. The example must use a feature-value
 * enconding. Usually, this enconding is determined by the dataset.
 * 
 * @author eraldof
 * 
 */
public interface DatasetExample {

	/**
	 * Return the dataset where this example lies in.
	 * 
	 * @return the dataset of this example.
	 */
	Corpus getDataset();

	/**
	 * Return the index of this example within its dataset.
	 * 
	 * @return the index of this example.
	 */
	int getIndex();

	/**
	 * Return the example ID.
	 * 
	 * @return the example ID.
	 */
	String getID();

	/**
	 * Return the feature-value encoding mapping.
	 * 
	 * @return the feature-value mapping.
	 */
	FeatureValueEncoding getFeatureEncoding();

	/**
	 * Test if the given feature-value is present in the given token.
	 * 
	 * @param token
	 *            the index of the token to be verified.
	 * @param feature
	 *            the index of the feature to be verified.
	 * @param value
	 *            the code of the value to be tested.
	 * 
	 * @return <code>true</code> if the feature in the given token has the given
	 *         value.
	 */
	boolean containFeatureValue(int token, int feature, int value);

	/**
	 * Set the value of a feature in a token.
	 * 
	 * @param token
	 *            the index of the token.
	 * @param feature
	 *            the index of the feature.
	 * @param value
	 *            the value to be set.
	 */
	void setFeatureValue(int token, int feature, int value);

	/**
	 * Set the value of a feature in a token.
	 * 
	 * Different of the previous method, this method accepts the feature and the
	 * feature value as string. So, it needs to encode theses values before
	 * setting it.
	 * 
	 * @param token
	 *            the index of the token.
	 * @param feature
	 *            the label of the feature.
	 * @param value
	 *            the value to be set.
	 */
	void setFeatureValue(int token, String feature, String value)
			throws DatasetException;

	/**
	 * Set the value of a feature in a token.
	 * 
	 * Different of the previous method, this method accepts the feature value
	 * as string. So, it needs to encode this value before setting it.
	 * 
	 * @param token
	 *            the index of the token.
	 * @param feature
	 *            the index of the feature.
	 * @param value
	 *            the value to be set.
	 */
	void setFeatureValue(int token, int feature, String value)
			throws DatasetException;

	/**
	 * Return the encoded value of a feature in a token.
	 * 
	 * @param token
	 *            the index of the token.
	 * @param feature
	 *            the index of the feature.
	 * 
	 * @return the encoded value of the feature in the token.
	 */
	int getFeatureValue(int token, int feature);

	/**
	 * Return the value (as a string) of a feature in a token.
	 * 
	 * @param token
	 *            the index of the token.
	 * @param feature
	 *            the index of the feature.
	 * 
	 * @return the string value of the feature in the token.
	 */
	String getFeatureValueAsString(int token, int feature);

	/**
	 * Return the value (as a string) of a feature in a token.
	 * 
	 * @param token
	 *            the index of the token.
	 * @param feature
	 *            the name of the feature.
	 * 
	 * @return the string value of the feature in the token.
	 */
	String getFeatureValueAsString(int token, String feature)
			throws DatasetException;

	/**
	 * Provide the number of tokens within this example.
	 * 
	 * @return the number of tokens within this example.
	 */
	int size();
}
