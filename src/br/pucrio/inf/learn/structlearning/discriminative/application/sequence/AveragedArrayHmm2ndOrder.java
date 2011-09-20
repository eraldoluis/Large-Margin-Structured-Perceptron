package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.Set;
import java.util.TreeSet;

/**
 * 2nd order HMM based on dense arrays and with support for voted perceptron.
 * 
 * @author eraldo
 * 
 */
public class AveragedArrayHmm2ndOrder extends Hmm2ndOrder {

	/**
	 * Number of states.
	 */
	private int numberOfStates;

	/**
	 * Number of symbols.
	 */
	private int numberOfSymbols;

	/**
	 * 2nd order transition parameters.
	 */
	private AveragedParameter[][][] transitions;

	/**
	 * Emission parameters. It is represented by a ordinary array (not sparse).
	 */
	private AveragedParameter[][] emissions;

	/**
	 * Set of parameters updated in the current iteration. Used to speedup the
	 * voted perceptron.
	 */
	private Set<AveragedParameter> updatedParameters;

	/**
	 * Initialize (alloc) an HMM with the given sizes.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 */
	public AveragedArrayHmm2ndOrder(int numberOfStates, int numberOfSymbols) {
		this.numberOfStates = numberOfStates;
		this.numberOfSymbols = numberOfSymbols;

		/*
		 * Allocate transition parameters arrays. We use additional dimensions
		 * to represent the initial state parameters (first token) and the
		 * initial transition parameters (from fist to second token).
		 */
		transitions = new AveragedParameter[numberOfStates + 1][numberOfStates + 1][numberOfStates];

		// Allocate emission parameters arrays.
		emissions = new AveragedParameter[numberOfStates][numberOfSymbols];

		// Allocate transition parameters.
		for (int state1 = 0; state1 < numberOfStates + 1; ++state1) {
			for (int state2 = 0; state2 < numberOfStates + 1; ++state2)
				for (int state3 = 0; state3 < numberOfStates; ++state3)
					transitions[state1][state2][state3] = new AveragedParameter();
		}

		// Allocate emission parameters.
		for (int state1 = 0; state1 < numberOfStates; ++state1)
			for (int symbol = 0; symbol < numberOfSymbols; ++symbol)
				emissions[state1][symbol] = new AveragedParameter();

		// Set of updated parameters within each iteration.
		this.updatedParameters = new TreeSet<AveragedParameter>();
	}

	@Override
	public int getNullState() {
		return numberOfStates;
	}

	@Override
	public int getNumberOfStates() {
		return numberOfStates;
	}

	@Override
	public int getNumberOfSymbols() {
		return numberOfSymbols;
	}

	@Override
	public double getTransitionParameter(int state1, int state2, int state3) {
		return transitions[state1][state2][state3].get();
	}

	@Override
	public double getEmissionParameter(int state, int symbol) {
		if (symbol < 0)
			return 0d;
		return emissions[state][symbol].get();
	}

	@Override
	public void setTransitionParameter(int state1, int state2, int state3,
			double value) {
		transitions[state1][state2][state3].set(value);
	}

	@Override
	public void setEmissionParameter(int state, int symbol, double value) {
		emissions[state][symbol].set(value);
	}

	@Override
	protected void updateTransitionParameter(int state1, int state2,
			int state3, double value) {
		transitions[state1][state2][state3].update(value);
		updatedParameters.add(transitions[state1][state2][state3]);
	}

	@Override
	protected void updateEmissionParameters(SequenceInput input, int token,
			int state, double value) {
		int numFtrs = input.getNumberOfInputFeatures(token);
		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr) {
			int ftr = input.getFeature(token, idxFtr);
			double weight = input.getFeatureWeight(token, idxFtr);
			emissions[state][ftr].update(value * weight);
			updatedParameters.add(emissions[state][ftr]);
		}
	}

	@Override
	public void sumUpdates(int iteration) {
		// Update the sum (used by the averaged-Perceptron) in each weight.
		for (AveragedParameter weight : updatedParameters)
			weight.sum(iteration);
		updatedParameters.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		// Average transition parameters.
		for (int state1 = 0; state1 < getNumberOfStates() + 1; ++state1) {
			for (int state2 = 0; state2 < getNumberOfStates() + 1; ++state2)
				for (int state3 = 0; state3 < getNumberOfStates(); ++state3)
					transitions[state1][state2][state3]
							.average(numberOfIterations);
		}

		// Average emission parameters.
		for (int state1 = 0; state1 < getNumberOfStates(); ++state1)
			for (int symbol = 0; symbol < emissions[state1].length; ++symbol)
				emissions[state1][symbol].average(numberOfIterations);
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		// Allocate an empty model.
		AveragedArrayHmm2ndOrder copy = new AveragedArrayHmm2ndOrder(
				getNumberOfStates(), emissions[0].length);

		// Clone transition parameters.
		for (int state1 = 0; state1 < getNumberOfStates() + 1; ++state1) {
			for (int state2 = 0; state2 < getNumberOfStates() + 1; ++state2)
				for (int state3 = 0; state3 < getNumberOfStates(); ++state3)
					copy.transitions[state1][state2][state3] = (AveragedParameter) transitions[state1][state2][state3]
							.clone();
		}

		// Clone emission parameters.
		for (int state1 = 0; state1 < getNumberOfStates(); ++state1)
			for (int symbol = 0; symbol < emissions[state1].length; ++symbol)
				copy.emissions[state1][symbol] = (AveragedParameter) emissions[state1][symbol]
						.clone();

		return copy;
	}

}
