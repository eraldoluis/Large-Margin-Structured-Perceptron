package br.pucrio.inf.learn.structlearning.application.sequence;

/**
 * Implementation of an HMM using primitive-type arrays to store the parameters.
 * 
 * @author eraldof
 * 
 */
public class ArrayBasedHmm extends Hmm {

	/**
	 * Model parameters: initial state weights. The array index is the state.
	 */
	private double[] initialState;

	/**
	 * Model parameters: state transition weights. The 2D-array index is
	 * comprised by the from-state and the to-state, respectively.
	 */
	private double[][] transitions;

	/**
	 * Model parameters: emission weights. The 2D-array index is composed by the
	 * state index and the symbol index, respectively.
	 */
	private double[][] emissions;

	/**
	 * Default state to choose when all states weight the same.
	 */
	private int defaultState;

	/**
	 * Initialize (alloc) an HMM with the given sizes.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 * @param defaultState
	 */
	public ArrayBasedHmm(int numberOfStates, int numberOfSymbols,
			int defaultState) {
		initialState = new double[numberOfStates];
		transitions = new double[numberOfStates][numberOfStates];
		emissions = new double[numberOfStates][numberOfSymbols];
		this.defaultState = defaultState;
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
		return initialState[state];
	}

	@Override
	public double getTransitionParameter(int fromState, int toState) {
		return transitions[fromState][toState];
	}

	@Override
	public double getEmissionParameter(int state, int symbol) {
		return emissions[state][symbol];
	}

	@Override
	protected void updateInitialStateParameter(int state, double value) {
		initialState[state] += value;
	}

	@Override
	protected void updateTransitionParameter(int fromToken, int toToken,
			double value) {
		transitions[fromToken][toToken] += value;
	}

	@Override
	protected void updateEmissionParameters(SequenceInput input, int token,
			int state, double value) {
		for (int ftr : input.getFeatures(token))
			emissions[state][ftr] += value;
	}

}
