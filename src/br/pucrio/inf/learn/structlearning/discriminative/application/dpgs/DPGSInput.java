package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.Serializable;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Input structure for dependency parsing with grandparent and sibling features.
 * 
 * This structure represents a sentence comprised by a sequence of tokens. Each
 * token comprises a fixed array (column-based dataset) of basic features. A
 * feature value can be a single integer or a list of integers (multi-valued
 * features).
 * 
 * Besides input features, this structure stores (grandparent and sibling)
 * factors of derived features, that is, features generated from templates.
 * Derived features are the features used by models to index parameters
 * (weights).
 * 
 * @author eraldo
 * 
 */
public class DPGSInput implements ExampleInput, Serializable {

	/**
	 * Auto-generated serial version ID.
	 */
	private static final long serialVersionUID = 3675190616084423770L;

	/**
	 * Index within the training dataset.
	 */
	private int trainingIndex;

	/**
	 * Textual ID of this example.
	 */
	private String id;

	/**
	 * Number of tokens in this sentence.
	 */
	private int numberOfTokens;

	/**
	 * Column-based features per token. Each token comprises a list of feature
	 * columns. Each feature value is an array of values. Usually, there is only
	 * one value in this array. But, for some features (multi-valued features),
	 * there are more values.
	 */
	private int[][][] basicFeatures;

	/**
	 * Derived features from grandparent templates. These templates involve
	 * three parameters: head, modifier and granparent (head of the head).
	 */
	private int[][][][] grandParentFeatures;

	/**
	 * Derived features from sibling templates. These templates involve three
	 * parameters: head, modifier and closest sibling before the modifier. These
	 * features comprise left and right sibling features. During feature
	 * generation, features are differentiated according to the relative
	 * position of modifiers in relation to the head token.
	 */
	private int[][][][] siblingsFeatures;

	/**
	 * Create a new grandparent/siblings input structure using the given feature
	 * arrays as underlying features. The given arrays are not copied, that is
	 * they are used as is and must not be modified after this method is called.
	 * 
	 * @param grandparentFeatures
	 * @param siblingsFeatures
	 * @throws DPGSException
	 */
	public DPGSInput(int[][][][] grandparentFeatures,
			int[][][][] siblingsFeatures) throws DPGSException {
		if (grandparentFeatures.length != siblingsFeatures.length)
			throw new DPGSException("Given grandparent feature array has "
					+ "different length of the siblings array");
		this.numberOfTokens = grandparentFeatures.length;
		this.grandParentFeatures = grandparentFeatures;
		this.siblingsFeatures = siblingsFeatures;
	}

	@Override
	public String getId() {
		return id;
	}

	@Override
	public int getTrainingIndex() {
		return trainingIndex;
	}

	/**
	 * Return the list of feature codes in the given grandparent factor.
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param idxGrandparent
	 * @return
	 */
	public int[] getGrandParentFeatures(int idxHead, int idxModifier,
			int idxGrandparent) {
		return grandParentFeatures[idxHead][idxModifier][idxGrandparent];
	}

	/**
	 * Return the list of feature codes in the given modifiers factor.
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param idxSibling
	 * @return
	 */
	public int[] getSiblingsFeatures(int idxHead, int idxModifier,
			int idxSibling) {
		return siblingsFeatures[idxHead][idxModifier][idxSibling];
	}

	@Override
	public DPGSOutput createOutput() {
		return new DPGSOutput(numberOfTokens);
	}

	@Override
	public void normalize(double norm) {
		throw new NotImplementedException();
	}

	@Override
	public void sortFeatures() {
		throw new NotImplementedException();
	}

	/**
	 * Return the number of token in this sentence.
	 * 
	 * @return
	 */
	public int size() {
		return numberOfTokens;
	}

}
