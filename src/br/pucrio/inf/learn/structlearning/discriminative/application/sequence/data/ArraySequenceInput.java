package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data;

import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

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
public class ArraySequenceInput implements SequenceInput {

	/**
	 * Identifier of this input example.
	 */
	private String id;

	/**
	 * Feature codes within this example. They are organized by tokens.
	 */
	private int[][] featureCodes;

	/**
	 * Feature values (weights) within this example. They are organized as the
	 * featureCodes properties.
	 */
	private double[][] featureWeights;

	/**
	 * Create a new sequence using the given ID and the given list of feature
	 * codes.
	 * 
	 * The feature weights are assumed to be one for features present in the
	 * list and zero otherwise.
	 * 
	 * @param id
	 * @param tokens
	 */
	public ArraySequenceInput(String id,
			Collection<? extends Collection<Integer>> tokens) {
		this.id = id;
		this.featureCodes = new int[tokens.size()][];
		this.featureWeights = new double[tokens.size()][];
		int tknIdx = 0;
		for (Collection<Integer> token : tokens) {
			this.featureCodes[tknIdx] = new int[token.size()];
			this.featureWeights[tknIdx] = new double[token.size()];

			int ftrIdx = 0;
			for (int ftr : token) {
				this.featureCodes[tknIdx][ftrIdx] = ftr;
				this.featureWeights[tknIdx][ftrIdx] = 1d;
				++ftrIdx;
			}

			++tknIdx;
		}
	}

	@Override
	public int size() {
		return featureCodes.length;
	}

	@Override
	public int getNumberOfInputFeatures(int token) {
		return featureCodes[token].length;
	}

	@Override
	public int getFeature(int token, int index) {
		return featureCodes[token][index];
	}

	@Override
	public double getFeatureWeight(int token, int index) {
		return featureWeights[token][index];
	}

	@Override
	public Iterable<Integer> getFeatureCodes(int token) {
		return new FeatureCodeIterator(token);
	}

	@Override
	public void normalize(double norm) {
		for (int tkn = 0; tkn < featureWeights.length; ++tkn) {
			// Current token weight vector.
			double[] weights = featureWeights[tkn];
			// Sum the weights.
			double sum = 0d;
			for (int ftr = 0; ftr < weights.length; ++ftr)
				sum += weights[ftr];
			// Normalize the weights.
			for (int ftr = 0; ftr < weights.length; ++ftr)
				weights[ftr] = weights[ftr] * norm / sum;
		}
	}

	@Override
	public ExampleOutput createOutput() {
		return new ArraySequenceOutput(featureCodes.length);
	}

	@Override
	public String getId() {
		return id;
	}

	/**
	 * Iterate over the features of a token.
	 * 
	 * @author eraldo
	 * 
	 */
	private class FeatureCodeIterator implements Iterator<Integer>,
			Iterable<Integer> {

		/**
		 * Token index whose features this iterator iterates over.
		 */
		private int token;

		/**
		 * Current index within the feature array.
		 */
		private int curIndex;

		/**
		 * Create an iterator over the features of the given token.
		 * 
		 * @param token
		 */
		public FeatureCodeIterator(int token) {
			this.token = token;
			this.curIndex = -1;
		}

		@Override
		public boolean hasNext() {
			return curIndex < featureCodes[token].length - 1;
		}

		@Override
		public Integer next() {
			++curIndex;
			return featureCodes[token][curIndex];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
					"This is an immutable iterator. One cannot remove an item.");
		}

		@Override
		public Iterator<Integer> iterator() {
			return this;
		}

	}

	@Override
	public void sortFeatureValues() {
		FeatureCodeComparator comp = new FeatureCodeComparator(null);
		for (int tkn = 0; tkn < featureCodes.length; ++tkn) {
			int numFtrs = featureCodes[tkn].length;

			// Create index array.
			Integer[] indexes = new Integer[numFtrs];
			for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr)
				indexes[idxFtr] = idxFtr;

			// Sort index array based on feature codes.
			comp.featureCodes = featureCodes[tkn];
			Arrays.sort(indexes, comp);

			// Sort codes and weights arrays according to the index array.
			int[] codes = new int[numFtrs];
			double[] weights = new double[numFtrs];
			for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr) {
				int idx = indexes[idxFtr].intValue();
				codes[idxFtr] = featureCodes[tkn][idx];
				weights[idxFtr] = featureWeights[tkn][idx];
			}

			// Exchange old arrays for the sorted ones.
			featureCodes[tkn] = codes;
			featureWeights[tkn] = weights;
		}
	}

	/**
	 * Comparator of feature codes based on an array of indexes.
	 * 
	 * @author eraldo
	 * 
	 */
	private static class FeatureCodeComparator implements Comparator<Integer> {
		/**
		 * Base array of feature codes.
		 */
		private int[] featureCodes;

		public FeatureCodeComparator(int[] featureCodes) {
			this.featureCodes = featureCodes;
		}

		@Override
		public int compare(Integer idx1, Integer idx2) {
			int i1 = idx1.intValue();
			int i2 = idx2.intValue();
			if (featureCodes[i1] < featureCodes[i2])
				return -1;
			if (featureCodes[i1] > featureCodes[i2])
				return 1;
			return 0;
		}
	}
}
