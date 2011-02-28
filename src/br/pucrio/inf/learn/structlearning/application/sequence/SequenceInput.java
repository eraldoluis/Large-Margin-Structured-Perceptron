package br.pucrio.inf.learn.structlearning.application.sequence;

import java.util.Collection;
import java.util.Iterator;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;

/**
 * Sequence of tokens. Each token comprises an array of features.
 * 
 * @author eraldo
 * 
 */
public class SequenceInput implements ExampleInput {

	/**
	 * Feature values for the tokens.
	 */
	private int[][] tokens;

	public SequenceInput(Collection<? extends Collection<Integer>> tokens) {
		this.tokens = new int[tokens.size()][];
		int tknIdx = 0;
		for (Collection<Integer> token : tokens) {
			this.tokens[tknIdx] = new int[token.size()];

			int ftrIdx = 0;
			for (int ftr : token) {
				this.tokens[tknIdx][ftrIdx] = ftr;
				++ftrIdx;
			}

			++tknIdx;
		}
	}

	/**
	 * Return the number of tokens in this sequence.
	 * 
	 * @return
	 */
	public int size() {
		return tokens.length;
	}

	/**
	 * Return the number of features for the given token.
	 * 
	 * @param token
	 * @return
	 */
	public int getNumberOfFeatures(int token) {
		return tokens[token].length;
	}

	/**
	 * Return the feature in the given index for the given token.
	 * 
	 * @param token
	 * @param index
	 * @return
	 */
	public int getFeature(int token, int index) {
		return tokens[token][index];
	}

	/**
	 * Return an iterator for the features of the given token.
	 * 
	 * @param token
	 * @return
	 */
	public Iterable<Integer> getFeatures(int token) {
		return new FeatureIterator(token);
	}

	/**
	 * Iterate over the features of a token.
	 * 
	 * @author eraldo
	 * 
	 */
	private class FeatureIterator implements Iterator<Integer>,
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
		public FeatureIterator(int token) {
			this.token = token;
			this.curIndex = -1;
		}

		@Override
		public boolean hasNext() {
			return curIndex < tokens[token].length - 1;
		}

		@Override
		public Integer next() {
			++curIndex;
			return tokens[token][curIndex];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
					"This is a immutable iterator. One cannot remove an item.");
		}

		@Override
		public Iterator<Integer> iterator() {
			return this;
		}

	}

}
