package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;

/**
 * Implementation of an HMM using averaged-weight maps to store the emission
 * parameters. For initial state and transition parameters, since they are in
 * small number, use ordinary arrays. This class is useful for the averaged
 * Perceptron algorithm.
 * 
 * @author eraldof
 * 
 */
public class AveragedMapHmm extends Hmm implements Cloneable {

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
	private List<HashMap<Integer, AveragedParameter>> emissions;

	/**
	 * Set of weights updated in the current iteration. Used to speedup the
	 * averaged-Perceptron.
	 */
	private Set<AveragedParameter> updatedWeights;

	/**
	 * Number of symbols that this model can handle.
	 */
	private int numberOfSymbols;

	/**
	 * Initialize (alloc) an HMM with the given sizes.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 */
	public AveragedMapHmm(int numberOfStates, int numberOfSymbols) {
		this.numberOfSymbols = numberOfSymbols;
		// Allocate arrays.
		initialState = new AveragedParameter[numberOfStates];
		transitions = new AveragedParameter[numberOfStates][numberOfStates];
		emissions = new ArrayList<HashMap<Integer, AveragedParameter>>(
				numberOfStates);

		// Allocate individual averaged weights.
		for (int state = 0; state < numberOfStates; ++state) {
			emissions.add(new HashMap<Integer, AveragedParameter>());
			initialState[state] = new AveragedParameter();
			for (int toState = 0; toState < numberOfStates; ++toState)
				transitions[state][toState] = new AveragedParameter();
		}

		this.updatedWeights = new TreeSet<AveragedParameter>();
	}

	@Override
	public int getNumberOfStates() {
		return initialState.length;
	}

	@Override
	public int getNumberOfSymbols() {
		return numberOfSymbols;
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
		AveragedParameter param = emissions.get(state).get(symbol);
		if (param == null)
			return 0d;
		return param.get();
	}

	@Override
	public void setInitialStateParameter(int state, double value) {
		initialState[state].set(value);
	}

	@Override
	public void setTransitionParameter(int fromState, int toState, double value) {
		transitions[fromState][toState].set(value);
	}

	/**
	 * Peek a parameter from its corresponding map or, if it does not exist yet,
	 * create it, insert in the map and return it.
	 * 
	 * @param state
	 * @param symbol
	 * @return
	 */
	protected AveragedParameter getEmissionAveragedParameter(int state,
			int symbol) {
		HashMap<Integer, AveragedParameter> map = emissions.get(state);
		AveragedParameter param = map.get(symbol);
		if (param == null) {
			param = new AveragedParameter();
			map.put(symbol, param);
		}
		return param;
	}

	@Override
	public void setEmissionParameter(int state, int symbol, double value) {
		getEmissionAveragedParameter(state, symbol).set(value);
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
			AveragedParameter param = getEmissionAveragedParameter(state, ftr);
			param.update(learningRate * weight);
			updatedWeights.add(param);
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
			for (AveragedParameter param : emissions.get(state).values())
				param.average(numberOfIterations);
		}
	}

	@Override
	public AveragedMapHmm clone() throws CloneNotSupportedException {
		// Allocate an empty model.
		AveragedMapHmm copy = new AveragedMapHmm(getNumberOfStates(),
				numberOfSymbols);

		// Clone each weight.
		for (int state = 0; state < getNumberOfStates(); ++state) {
			copy.initialState[state] = (AveragedParameter) initialState[state]
					.clone();
			for (int toState = 0; toState < getNumberOfStates(); ++toState)
				copy.transitions[state][toState] = (AveragedParameter) transitions[state][toState]
						.clone();
			// Emission parameter maps.
			HashMap<Integer, AveragedParameter> clonedMap = copy.emissions
					.get(state);
			for (Entry<Integer, AveragedParameter> paramEntry : emissions.get(
					state).entrySet())
				clonedMap.put(paramEntry.getKey(), paramEntry.getValue()
						.clone());
		}

		return copy;
	}
}
