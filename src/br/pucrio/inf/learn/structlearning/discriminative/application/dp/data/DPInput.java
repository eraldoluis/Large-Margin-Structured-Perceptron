package br.pucrio.inf.learn.structlearning.discriminative.application.dp.data;

import java.io.Serializable;
import java.util.Collection;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Input structure of a dependency parsing example. Represent a complete
 * directed graph whose nodes are tokens of a sentence. An edge is composed by a
 * list of features between two tokens in the sentence.
 * 
 * @author eraldo
 * 
 */
public class DPInput implements ExampleInput, Serializable {

	/**
	 * Automatically generated serial version id.
	 */
	private static final long serialVersionUID = -2050169475499863867L;

	/**
	 * Number of tokens in this input structure.
	 */
	private int numberOfTokens;

	/**
	 * List of features for each pair of tokens (edge). The diagonal
	 * (self-loops) can be ignored. It is a matrix of arrays of feature codes.
	 */
	private int[][][] features;

	/**
	 * For template-based models, features are represented by basic features
	 * that are combined to derive composed features.
	 */
	private int[][][] basicFeatures;

	/**
	 * For template evolution training, feature weights of templates from
	 * previous partitions are fixed and the corresponding weights are stored
	 * here.
	 */
	private double[][] fixedWeights;

	/**
	 * Indicate which tokens are tagged as punctuation.
	 */
	private boolean[] punctuation;

	/**
	 * Identification of the example.
	 */
	private String id;

	/**
	 * Index of this example within the array of traning examples, when this
	 * example is part of a training set.
	 */
	private int trainingIndex;

	/**
	 * Create the input structure of a training example.
	 * 
	 * @param trainingIndex
	 * @param featuresCollection
	 * @throws DPInputException
	 */
	public DPInput(
			int trainingIndex,
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection)
			throws DPInputException {
		this.trainingIndex = trainingIndex;
		this.id = id;
		allocAndCopyFeatures(featuresCollection);
	}

	/**
	 * Create the input structure of an example that can represent a test
	 * example or just an input structure whose output structure need to be
	 * predicted.
	 * 
	 * @param featuresCollection
	 * @throws DPInputException
	 */
	public DPInput(
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection)
			throws DPInputException {
		this.trainingIndex = -1;
		this.id = id;
		allocAndCopyFeatures(featuresCollection);
	}

	/**
	 * Create the input structure of an example. The given features are basic
	 * features, that is they come from a column-based dataset.
	 * 
	 * @param id
	 * @param basicFeaturesCollection
	 * @param allocFixedWeightsMatrix
	 * @throws DPInputException
	 */
	public DPInput(
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> basicFeaturesCollection,
			boolean allocFixedWeightsMatrix) throws DPInputException {
		this.id = id;
		this.trainingIndex = -1;
		allocAndCopyBasicFeatures(basicFeaturesCollection,
				allocFixedWeightsMatrix);
	}

	/**
	 * @return the number of tokens in this example.
	 */
	public int getNumberOfTokens() {
		return numberOfTokens;
	}

	/**
	 * Set the flags of punctuation for each token.
	 * 
	 * @param vals
	 */
	public void setPunctuation(boolean[] vals) {
		punctuation = vals;
	}

	/**
	 * Return <code>true</code> if the given token is tagged as punctuation.
	 * 
	 * @param token
	 * @return
	 */
	public boolean isPunctuation(int token) {
		return punctuation[token];
	}

	/**
	 * Return the fixed weight associated with the given edge.
	 * 
	 * @param idxHead
	 * @param idxDependent
	 * @return
	 */
	public double getFixedWeight(int idxHead, int idxDependent) {
		if (fixedWeights == null)
			return 0d;
		return fixedWeights[idxHead][idxDependent];
	}

	@Override
	public String getId() {
		return id;
	}

	@Override
	public int getTrainingIndex() {
		return trainingIndex;
	}

	@Override
	public DPOutput createOutput() {
		return new DPOutput(numberOfTokens);
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
	 * Return list of feature codes for the given edge (pair of tokens).
	 * 
	 * @param idxHead
	 * @param idxDep
	 * @return
	 */
	public int[] getFeatures(int idxHead, int idxDep) {
		return features[idxHead][idxDep];
	}

	/**
	 * Return the vector of basic features codes.
	 * 
	 * @param idxHead
	 * @param idxDep
	 * @return
	 */
	public int[] getBasicFeatures(int idxHead, int idxDep) {
		return basicFeatures[idxHead][idxDep];
	}

	/**
	 * Allocate feature matrix for this input structure.
	 */
	public void allocFeatureMatrix() {
		features = new int[numberOfTokens][numberOfTokens][];
	}

	/**
	 * Set the explicit features of the given edge.
	 * 
	 * @param idxHead
	 * @param idxDep
	 * @param itFeatures
	 * @param size
	 */
	public void setFeatures(int idxHead, int idxDep,
			Iterable<Integer> itFeatures, int size) {
		int[] edgeFeatures = new int[size];
		features[idxHead][idxDep] = edgeFeatures;
		int idxFtr = 0;
		for (int ftr : itFeatures)
			edgeFeatures[idxFtr++] = ftr;
	}

	/**
	 * Allocate the matrix of feature codes and fill it with the values in the
	 * given collection of collections. The given matrix is tranposed, i.e.,
	 * lines are destin tokens and columns are origin tokens. The first line of
	 * the matrix is ignored since there is no point in considering edges
	 * entering the root token (zero), i.e., the root token is never a dependent
	 * token.
	 * 
	 * @param featuresCollection
	 * @throws DPInputException
	 *             if the number of columns in any line is different of the
	 *             number of lines.
	 */
	private void allocAndCopyFeatures(
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection)
			throws DPInputException {
		/*
		 * Number of tokens. The first line is ommited since there is no point
		 * in considering edges entering the root token.
		 */
		numberOfTokens = featuresCollection.size();

		// Allocate feature matrix (a matrix of arrays of feature codes).
		features = new int[numberOfTokens][numberOfTokens][];

		// Copy feature values from collection to matrix.
		copyFeatures(featuresCollection, features);
	}

	/**
	 * Allocate the basic features matrix and copy the given values to it.
	 * Optionally, also allocate the fixed weights matrix.
	 * 
	 * @param featuresCollection
	 * @param allocFixedWeightsMatrix
	 * @throws DPInputException
	 */
	private void allocAndCopyBasicFeatures(
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection,
			boolean allocFixedWeightsMatrix) throws DPInputException {
		// Number of tokens in this input.
		numberOfTokens = featuresCollection.size();

		// Allocate feature matrix (a matrix of arrays of feature codes).
		basicFeatures = new int[numberOfTokens][numberOfTokens][];

		if (allocFixedWeightsMatrix)
			// Allocate fixed weights matrix.
			fixedWeights = new double[numberOfTokens][numberOfTokens];

		// Copy feature values from collection to matrix.
		copyFeatures(featuresCollection, basicFeatures);
	}

	/**
	 * Copy the feature values from a source collection to a target matrix.
	 * 
	 * @param sourceFeaturesCollection
	 * @param targetFeaturesMatrix
	 * @throws DPInputException
	 */
	private void copyFeatures(
			Collection<? extends Collection<? extends Collection<Integer>>> sourceFeaturesCollection,
			int[][][] targetFeaturesMatrix) throws DPInputException {
		// Number of tokens in this input.
		numberOfTokens = targetFeaturesMatrix.length;

		// No features to token zero.
		for (int from = 0; from < numberOfTokens; ++from)
			targetFeaturesMatrix[from][0] = null;

		int idxTokenTo = 0;
		for (Collection<? extends Collection<Integer>> tokensFrom : sourceFeaturesCollection) {
			if (idxTokenTo == 0) {
				// Root token is never a dependent.
				++idxTokenTo;
				continue;
			}

			// Number of lines must be the same number of columns.
			if (tokensFrom.size() != numberOfTokens)
				throw new DPInputException(
						"Input matrix is not square. Number of lines is "
								+ numberOfTokens
								+ " but number of columns in line "
								+ idxTokenTo + " is " + tokensFrom.size());

			int idxTokenFrom = 0;
			for (Collection<Integer> tokenFrom : tokensFrom) {
				// Skip the diagonal elements.
				if (tokenFrom != null) {
					// Allocate the array of feature codes.
					targetFeaturesMatrix[idxTokenFrom][idxTokenTo] = new int[tokenFrom
							.size()];
					int idxFtr = 0;
					for (int ftrCode : tokenFrom) {
						targetFeaturesMatrix[idxTokenFrom][idxTokenTo][idxFtr] = ftrCode;
						++idxFtr;
					}
				} else
					targetFeaturesMatrix[idxTokenFrom][idxTokenTo] = null;

				++idxTokenFrom;
			}

			++idxTokenTo;
		}
	}

}
