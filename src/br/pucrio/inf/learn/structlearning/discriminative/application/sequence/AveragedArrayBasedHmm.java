package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.Set;
import java.util.TreeSet;

/**
 * Implementation of an HMM using averaged-weight arrays to store the
 * parameters. This class is useful for the averaged Perceptron algorithm.
 * 
 * @author eraldof
 * 
 */
public class AveragedArrayBasedHmm extends Hmm implements Cloneable {

	/**
	 * Model parameters: initial state weights. The array index is the state.
	 */
	private AveragedWeight[] initialState;

	/**
	 * Model parameters: state transition weights. The 2D-array index is
	 * comprised by the from-state and the to-state, respectively.
	 */
	private AveragedWeight[][] transitions;

	/**
	 * Model parameters: emission weights. The 2D-array index is composed by the
	 * state index and the symbol index, respectively.
	 */
	private AveragedWeight[][] emissions;

	/**
	 * Set of weights updated in the current iteration. Used to speedup the
	 * averaged-Perceptron.
	 */
	private Set<AveragedWeight> updatedWeights;

	/**
	 * Initialize (alloc) an HMM with the given sizes.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 */
	public AveragedArrayBasedHmm(int numberOfStates, int numberOfSymbols) {
		// Allocate arrays.
		initialState = new AveragedWeight[numberOfStates];
		transitions = new AveragedWeight[numberOfStates][numberOfStates];
		emissions = new AveragedWeight[numberOfStates][numberOfSymbols];

		// Allocate individual averaged weights.
		for (int state = 0; state < numberOfStates; ++state) {
			initialState[state] = new AveragedWeight();
			for (int toState = 0; toState < numberOfStates; ++toState)
				transitions[state][toState] = new AveragedWeight();
			for (int symbol = 0; symbol < numberOfSymbols; ++symbol)
				emissions[state][symbol] = new AveragedWeight();
		}

		this.updatedWeights = new TreeSet<AveragedWeight>();
	}

	@Override
	public int getNumberOfStates() {
		return initialState.length;
	}

	@Override
	public int getNumberOfSymbols() {
		if (emissions.length == 0)
			return 0;
		// This data structure explicitly represents all features for all state.
		return emissions[0].length;
	}

	@Override
	public double getInitialStateParameter(int state) {
		return initialState[state].get();
	}

	@Override
	public double getTransitionParameter(int fromState, int toState) {
		return transitions[fromState][toState].get();
	}

	@Override
	public double getEmissionParameter(int state, int symbol) {
		if (symbol < 0)
			return 0d;
		return emissions[state][symbol].get();
	}

	@Override
	public void setInitialStateParameter(int state, double value) {
		initialState[state].set(value);
	}

	@Override
	public void setTransitionParameter(int fromState, int toState, double value) {
		transitions[fromState][toState].set(value);
	}

	@Override
	public void setEmissionParameter(int state, int symbol, double value) {
		emissions[state][symbol].set(value);
	}

	@Override
	protected void updateInitialStateParameter(int state, double value) {
		initialState[state].update(value);
		updatedWeights.add(initialState[state]);
	}

	@Override
	protected void updateTransitionParameter(int fromToken, int toToken,
			double learningRate) {
		transitions[fromToken][toToken].update(learningRate);
		updatedWeights.add(transitions[fromToken][toToken]);
	}

	@Override
	protected void updateEmissionParameters(SequenceInput input, int token,
			int state, double learningRate) {
		int numFtrs = input.getNumberOfFeatures(token);
		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr) {
			int ftr = input.getFeature(token, idxFtr);
			double weight = input.getFeatureWeight(token, idxFtr);
			emissions[state][ftr].update(learningRate * weight);
			updatedWeights.add(emissions[state][ftr]);
		}
	}

	@Override
	public void sumUpdates(int iteration) {
		// Update the sum (used by the averaged-Perceptron) in each weight.
		for (AveragedWeight weight : updatedWeights)
			weight.sum(iteration);
		updatedWeights.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		// Average all the weights.
		for (int state = 0; state < getNumberOfStates(); ++state) {
			initialState[state].average(numberOfIterations);
			for (int toState = 0; toState < getNumberOfStates(); ++toState)
				transitions[state][toState].average(numberOfIterations);
			for (int symbol = 0; symbol < emissions[state].length; ++symbol)
				emissions[state][symbol].average(numberOfIterations);
		}
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		// Allocate an empty model.
		AveragedArrayBasedHmm copy = new AveragedArrayBasedHmm(
				getNumberOfStates(), emissions[0].length);

		// Clone each weight.
		for (int state = 0; state < getNumberOfStates(); ++state) {
			copy.initialState[state] = (AveragedWeight) initialState[state]
					.clone();
			for (int toState = 0; toState < getNumberOfStates(); ++toState)
				copy.transitions[state][toState] = (AveragedWeight) transitions[state][toState]
						.clone();
			for (int symbol = 0; symbol < emissions[state].length; ++symbol)
				copy.emissions[state][symbol] = (AveragedWeight) emissions[state][symbol]
						.clone();
		}

		return copy;
	}

	/**
	 * Weight that supports an averaged-Perceptron implementation.
	 * 
	 * @author eraldof
	 * 
	 */
	private static class AveragedWeight implements Comparable<AveragedWeight>,
			Cloneable {
		/**
		 * The current (non-averaged) weight. This value must be used by the
		 * inference algorithm through the Perceptron execution.
		 */
		private double weight;

		/**
		 * Update realized within the current iteration. This must be summed to
		 * the <code>sum</code> value at the end of each iteration.
		 */
		private double update;

		/**
		 * The current sum of the values assumed by this weight in all previous
		 * iterations.
		 */
		private double sum;

		/**
		 * Last iteration when this weight was summed (<code>update</code> value
		 * was summed into the <code>sum</code> value).
		 */
		private int lastSummedIteration;

		/**
		 * Set the value of this weight.
		 * 
		 * @param value
		 */
		public void set(double value) {
			weight = value;
			sum = 0d;
			update = 0d;
		}

		/**
		 * Add the given value <code>val</code> to this weight. In fact, this
		 * value is added to the <code>update</code> before being incorporated
		 * in the weight itself.
		 * 
		 * @param val
		 */
		public void update(double val) {
			update += val;
		}

		/**
		 * Return the current value of this weight.
		 * 
		 * @return
		 */
		public double get() {
			return weight;
		}

		/**
		 * Account the last updates in its weight and in its summed (for later
		 * averaging) value.
		 * 
		 * @param iteration
		 */
		public void sum(int iteration) {
			sum += weight * (iteration - lastSummedIteration) + update;
			weight += update;
			update = 0d;
			lastSummedIteration = iteration;
		}

		/**
		 * Average this weight.
		 * 
		 * @param numberOfIterations
		 */
		public void average(int numberOfIterations) {
			// Account any residual value.
			sum(numberOfIterations - 1);
			// Average.
			weight = sum / numberOfIterations;
			// Keep track that this weight was already averaged.
			sum = Double.NEGATIVE_INFINITY;
		}

		@Override
		public int compareTo(AveragedWeight other) {
			return toString().compareTo(other.toString());
		}

		@Override
		protected Object clone() throws CloneNotSupportedException {
			return super.clone();
		}

	}

}
