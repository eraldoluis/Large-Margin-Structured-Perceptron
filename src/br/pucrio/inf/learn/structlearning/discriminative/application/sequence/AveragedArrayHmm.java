package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.Set;
import java.util.TreeSet;

import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;

/**
 * Implementation of an HMM using averaged-weight arrays to store the
 * parameters. This class is useful for the averaged Perceptron algorithm.
 * 
 * @author eraldof
 * 
 */
public class AveragedArrayHmm extends Hmm implements Cloneable {

	/**
	 * Model parameters: initial state weights. The array index is the state.
	 */
	private AveragedParameter[] initialState;

	/**
	 * Model parameters: state transition weights. The 2D-array index is
	 * comprised by the from-state and the to-state, respectively.
	 */
	private AveragedParameter[][] transitions;

	/**
	 * Model parameters: emission weights. The 2D-array index is composed by the
	 * state index and the symbol index, respectively.
	 */
	private AveragedParameter[][] emissions;

	/**
	 * Set of weights updated in the current iteration. Used to speedup the
	 * averaged-Perceptron.
	 */
	private Set<AveragedParameter> updatedWeights;

	/**
	 * Initialize (alloc) an HMM with the given sizes.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 */
	public AveragedArrayHmm(int numberOfStates, int numberOfSymbols) {
		// Allocate arrays.
		initialState = new AveragedParameter[numberOfStates];
		transitions = new AveragedParameter[numberOfStates][numberOfStates];
		emissions = new AveragedParameter[numberOfStates][numberOfSymbols];

		// Allocate individual averaged weights.
		for (int state = 0; state < numberOfStates; ++state) {
			initialState[state] = new AveragedParameter();
			for (int toState = 0; toState < numberOfStates; ++toState)
				transitions[state][toState] = new AveragedParameter();
			for (int symbol = 0; symbol < numberOfSymbols; ++symbol)
				emissions[state][symbol] = new AveragedParameter();
		}

		this.updatedWeights = new TreeSet<AveragedParameter>();
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
	protected void updateTransitionParameter(int fromState, int toState,
			double value) {
		transitions[fromState][toState].update(value);
		updatedWeights.add(transitions[fromState][toState]);
	}

	@Override
	protected void updateEmissionParameters(SequenceInput input, int token,
			int state, double learningRate) {
		int numFtrs = input.getNumberOfInputFeatures(token);
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
		for (AveragedParameter weight : updatedWeights)
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
	public AveragedArrayHmm clone() throws CloneNotSupportedException {
		// Allocate an empty model.
		AveragedArrayHmm copy = new AveragedArrayHmm(getNumberOfStates(),
				emissions[0].length);

		// Clone each weight.
		for (int state = 0; state < getNumberOfStates(); ++state) {
			copy.initialState[state] = (AveragedParameter) initialState[state]
					.clone();
			for (int toState = 0; toState < getNumberOfStates(); ++toState)
				copy.transitions[state][toState] = (AveragedParameter) transitions[state][toState]
						.clone();
			for (int symbol = 0; symbol < emissions[state].length; ++symbol)
				copy.emissions[state][symbol] = (AveragedParameter) emissions[state][symbol]
						.clone();
		}

		return copy;
	}

}
