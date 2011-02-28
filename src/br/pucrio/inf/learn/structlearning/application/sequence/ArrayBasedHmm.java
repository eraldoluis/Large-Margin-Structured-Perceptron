package br.pucrio.inf.learn.structlearning.application.sequence;

import java.util.Set;
import java.util.TreeSet;

/**
 * Implementation of an HMM using primitive-type arrays to store the parameters.
 * This implementation also supports the averaged perceptron.
 * 
 * @author eraldof
 * 
 */
public class ArrayBasedHmm extends Hmm {

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
	 * Default state to choose when all states weight the same.
	 */
	private int defaultState;

	private Set<AveragedWeight> updatedWeights;

	/**
	 * Initialize (alloc) an HMM with the given sizes.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 * @param defaultState
	 */
	public ArrayBasedHmm(int numberOfStates, int numberOfSymbols,
			int defaultState) {
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

		this.defaultState = defaultState;
		this.updatedWeights = new TreeSet<AveragedWeight>();
	}

	@Override
	public int getNumberOfStates() {
		return initialState.length;
	}

	@Override
	protected int getDefaultState() {
		return defaultState;
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
		return emissions[state][symbol].get();
	}

	@Override
	protected void updateInitialStateParameter(int state, double value) {
		initialState[state].update(value);
		updatedWeights.add(initialState[state]);
	}

	@Override
	protected void updateTransitionParameter(int fromToken, int toToken,
			double value) {
		transitions[fromToken][toToken].update(value);
		updatedWeights.add(transitions[fromToken][toToken]);
	}

	@Override
	protected void updateEmissionParameters(SequenceInput input, int token,
			int state, double value) {
		for (int ftr : input.getFeatures(token)) {
			emissions[state][ftr].update(value);
			updatedWeights.add(emissions[state][ftr]);
		}
	}

	@Override
	public void sumAfterIteration(int iteration) {
		for (AveragedWeight weight : updatedWeights)
			weight.sum(iteration);
		updatedWeights.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		for (int state = 0; state < getNumberOfStates(); ++state) {
			initialState[state].average(numberOfIterations);
			for (int toState = 0; toState < getNumberOfStates(); ++toState)
				transitions[state][toState].average(numberOfIterations);
			for (int symbol = 0; symbol < emissions[state].length; ++symbol)
				emissions[state][symbol].average(numberOfIterations);
		}
	}

	/**
	 * Weight that supports an averaged-Perceptron implementation.
	 * 
	 * @author eraldof
	 * 
	 */
	private static class AveragedWeight {
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

		public void update(double val) {
			update += val;
		}

		public double get() {
			return weight;
		}

		public void sum(int iteration) {
			sum = sum * (iteration - lastSummedIteration) + update;
			weight += update;
			update = 0d;
			lastSummedIteration = iteration;
		}

		public void average(int numberOfIterations) {
			// Account any residual value.
			sum(numberOfIterations - 1);
			// Average.
			weight = sum / numberOfIterations;
			// Keep track that this weight was already averaged.
			sum = Double.NEGATIVE_INFINITY;
		}
	}

}
