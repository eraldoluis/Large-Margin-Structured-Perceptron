package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import java.util.Collection;
import java.util.Iterator;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Person-Quotation input structure.
 * 
 * @author eraldo
 * 
 */
public class PQInput implements ExampleInput {

	/**
	 * Identifier of the document that this example belongs to.
	 */
	private String docId;

	/**
	 * Index of this sequence input within the array of training examples, when
	 * it is a training example. Otherwise, this value is -1.
	 */
	private int trainingIndex;

	/**
	 * Sparse representation of binary features. This is an array of persons.
	 * Each person is a list of features describing the pair (quotation,person).
	 */
	private int[][] features;

	/**
	 * Create a new PQ input using the given docID and the given list of
	 * feature codes.
	 * 
	 * @param docId
	 * @param tokens
	 */
	public PQInput(String docId,
			Collection<? extends Collection<Integer>> tokens) {
		this.docId = docId;
		this.trainingIndex = -1;
		this.features = new int[tokens.size()][];

		int tknIdx = 0;
		for (Collection<Integer> token : tokens) {
			this.features[tknIdx] = new int[token.size()];

			int ftrIdx = 0;
			for (int ftr : token) {
				this.features[tknIdx][ftrIdx] = ftr;

				++ftrIdx;
			}

			++tknIdx;
		}
	}
	
	/**
	 * Create a new PQ input using the given docID and the given list of
	 * feature codes.
	 * 
	 * @param docId
	 * @param trainingIndex
	 * @param tokens
	 */
	public PQInput(String docId, int trainingIndex,
			Collection<? extends Collection<Integer>> tokens) {
		this(docId, tokens);
		this.trainingIndex = trainingIndex;
	}

	@Override
	public String getId() {
		return null;
	}

	@Override
	public PQOutput createOutput() {
		return new PQOutput();
	}

	@Override
	public void normalize(double norm) {
		// TODO only to normalize input vectors
	}

	@Override
	public void sortFeatures() {
		// TODO only to use kernel functions
	}

	@Override
	public int getTrainingIndex() {
		return -1;
	}

	public int size() {
		return features.length;
	}

	public Iterable<Integer> getFeatureCodes(int token) {
		return new FeatureCodeIterator(token);
	}

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
			return curIndex < features[token].length - 1;
		}

		@Override
		public Integer next() {
			++curIndex;
			return features[token][curIndex];
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

}
