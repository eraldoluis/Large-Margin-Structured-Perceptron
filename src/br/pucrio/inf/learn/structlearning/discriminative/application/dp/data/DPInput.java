package br.pucrio.inf.learn.structlearning.discriminative.application.dp.data;

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
public class DPInput implements ExampleInput {

	/**
	 * List of features for each pair of tokens (edge). The diagonal
	 * (self-loops) can be ignored. It is a matrix of arrays of feature codes.
	 */
	private int[][][] features;

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
	 * @return the number of tokens in this example.
	 */
	public int getNumberOfTokens() {
		return features.length;
	}

	/**
	 * Allocate the matrix of feature codes and fill it with the values in the
	 * given collection of collections. The given matrix is tranposed, i.e.,
	 * lines are destin tokens and columns are origin tokens. The first line of
	 * the matrix is ommited since there is not point in considering edges
	 * entering the root token (zero).
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
		int numTokens = featuresCollection.size() + 1;

		// Allocate feature matrix (a matrix of arrays of feature codes).
		features = new int[numTokens][numTokens][];

		// No features to token zero.
		for (int from = 0; from < numTokens; ++from)
			features[from][0] = new int[0];

		int idxTokenTo = 1;
		for (Collection<? extends Collection<Integer>> tokensFrom : featuresCollection) {
			// Number of lines must be the same number of columns.
			if (tokensFrom.size() != numTokens)
				throw new DPInputException(
						"Input matrix is not square. Number of lines is "
								+ numTokens + " but number of columns in line "
								+ idxTokenTo + " is " + tokensFrom.size());

			int idxTokenFrom = 0;
			for (Collection<Integer> tokenFrom : tokensFrom) {
				// Skip the diagonal elements.
				if (idxTokenFrom != idxTokenTo) {
					// Allocate the array of feature codes.
					features[idxTokenFrom][idxTokenTo] = new int[tokenFrom
							.size()];
					int idxFtr = 0;
					for (int ftrCode : tokenFrom) {
						features[idxTokenFrom][idxTokenTo][idxFtr] = ftrCode;
						++idxFtr;
					}
				} else
					features[idxTokenFrom][idxTokenTo] = new int[0];

				++idxTokenFrom;
			}

			++idxTokenTo;
		}
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
		return new DPOutput(features.length);
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
	 * @param head
	 * @param dependent
	 * @return
	 */
	public int[] getFeatureCodes(int head, int dependent) {
		return features[head][dependent];
	}

}
