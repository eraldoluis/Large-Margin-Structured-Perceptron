package br.pucrio.inf.learn.structlearning.application.sequence;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.structlearning.task.Inference;

/**
 * Implement Viterbi-based inference algorithms for sequence structures.
 * 
 * @author eraldof
 * 
 */
public class ViterbiInference implements Inference {

	/**
	 * Default state to be choosed when all states weight the same.
	 */
	private int defaultState;

	/**
	 * State code that indicates non-annotated tokens. In general, this will be
	 * an invalid value (less than zero or greater than the number of states).
	 */
	private int nonAnnotatedStateCode;

	/**
	 * Weight of the loss function in the objective function.
	 */
	private double lossWeight;

	/**
	 * This is the correct (or loss reference) output sequence corresponding to
	 * the current input sequence. This sequence is used to calculate the loss
	 * function.
	 */
	private SequenceOutput lossReferenceOutput;

	/**
	 * Create a Viterbi inference algorithm using the given state as the default
	 * option for every token.
	 * 
	 * @param defaultState
	 */
	public ViterbiInference(int defaultState) {
		this.defaultState = defaultState;
		this.nonAnnotatedStateCode = -1;
		this.lossWeight = 0d;
		this.lossReferenceOutput = null;
	}

	/**
	 * Create a Viterbi inference algorithm using the given state as the default
	 * state and the given code as the indicator of non-annotated tokens, i.e.,
	 * token labeled with this state code are considered unlabeled.
	 * 
	 * @param defaultState
	 * @param nonAnnotatedStateCode
	 */
	public ViterbiInference(int defaultState, int nonAnnotatedStateCode) {
		this(defaultState);
		this.nonAnnotatedStateCode = nonAnnotatedStateCode;
	}

	/**
	 * Set the state code that indicates non-annotated tokens. This might be an
	 * invalid value, i.e., less than zero or greater than the number of states.
	 * 
	 * @param nonAnnotatedStateCode
	 */
	public void setNonAnnotatedStateCode(int nonAnnotatedStateCode) {
		this.nonAnnotatedStateCode = nonAnnotatedStateCode;
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		tag((Hmm) model, (SequenceInput) input, (SequenceOutput) output);
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		partialTag((Hmm) model, (SequenceInput) input,
				(SequenceOutput) partiallyLabeledOutput,
				(SequenceOutput) predictedOutput);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput correctOutput, ExampleOutput inferedOutput,
			double lossWeight) {
		// Save the current configuration.
		double previousLossWeight = this.lossWeight;
		SequenceOutput previousLossReferenceOutput = this.lossReferenceOutput;

		// Configure the loss-augmented necessary properties.
		this.lossWeight = lossWeight;
		this.lossReferenceOutput = (SequenceOutput) correctOutput;

		// Call the ordinary inference algorithm.
		tag((Hmm) model, (SequenceInput) input, (SequenceOutput) inferedOutput);

		// Restore the previous configuration.
		this.lossWeight = previousLossWeight;
		this.lossReferenceOutput = previousLossReferenceOutput;
	}

	/**
	 * Tag the output with the best label sequence for the given input and HMM.
	 * 
	 * @param hmm
	 * @param input
	 * @param output
	 */
	public void tag(Hmm hmm, SequenceInput input, SequenceOutput output) {
		// Example length.
		int numberOfStates = hmm.getNumberOfStates();
		int lenExample = input.size();

		// Best partial-path weights.
		double[][] delta = new double[lenExample][numberOfStates];
		// Best partial-path backward table.
		int[][] psi = new int[lenExample][numberOfStates];

		// The default state is always the fisrt option.
		int bestState = defaultState;
		double bestWeight = delta[lenExample - 1][defaultState];

		// Weights for the first token.
		for (int state = 0; state < numberOfStates; ++state) {
			delta[0][state] = getTokenEmissionWeightWithLoss(hmm, input, 0,
					state) + hmm.getInitialStateParameter(state);
		}

		// Apply each step of the Viterbi algorithm.
		for (int tkn = 1; tkn < lenExample; ++tkn)
			for (int state = 0; state < numberOfStates; ++state)
				viterbi(hmm, delta, psi, input, tkn, state, defaultState);

		// The default state is always the fisrt option.
		bestState = defaultState;
		bestWeight = delta[lenExample - 1][defaultState];

		// Find the best last state.
		for (int state = 0; state < numberOfStates; ++state) {
			double weight = delta[lenExample - 1][state];
			if (weight > bestWeight) {
				bestWeight = weight;
				bestState = state;
			}
		}

		// Reconstruct the best path from the best final state, and tag the
		// input.
		backwardTag(output, psi, bestState);
	}

	/**
	 * Calculate the best previous state (fromState) to the given end-state
	 * <code>toState</code> for the given token <code>token</code>.
	 * 
	 * @param hmm
	 *            the HMM to be used
	 * @param delta
	 *            contain the best accumulated weights until the previous token
	 * @param psi
	 *            used to store the best option
	 * @param input
	 *            the input structure
	 * @param token
	 *            the token be considered
	 * @param toState
	 *            the state to be considered
	 * @param defaultState
	 *            the default state
	 */
	protected void viterbi(Hmm hmm, double[][] delta, int[][] psi,
			SequenceInput input, int token, int toState, int defaultState) {
		// Number of states.
		int numStates = hmm.getNumberOfStates();

		// Choose the best previous state (consider only the transition weight).
		int maxState = defaultState;
		double maxWeight = delta[token - 1][defaultState]
				+ hmm.getTransitionParameter(defaultState, toState);
		for (int fromState = 0; fromState < numStates; ++fromState) {
			double weight = delta[token - 1][fromState]
					+ hmm.getTransitionParameter(fromState, toState);
			if (weight > maxWeight) {
				maxWeight = weight;
				maxState = fromState;
			}
		}

		// Set delta and psi according to the best from-state.
		psi[token][toState] = maxState;
		delta[token][toState] = maxWeight
				+ getTokenEmissionWeightWithLoss(hmm, input, token, toState);
	}

	/**
	 * Tag the given input sequence maintaining the annotations of some tokens.
	 * The annotations are given in the output sequence
	 * <code>partiallyLabeledOutput</code> by considering that non-annotated
	 * tokens are labeled (in this partially-labeled sequence) using a special
	 * state (that is set before). The resulting tag sequence is stored in the
	 * output sequence <code>predictedOutput</code>.
	 * 
	 * @param hmm
	 * @param input
	 * @param partiallyLabeledOutput
	 * @param predictedOutput
	 */
	public void partialTag(Hmm hmm, SequenceInput input,
			SequenceOutput partiallyLabeledOutput,
			SequenceOutput predictedOutput) {
		// Example length.
		int numberOfStates = hmm.getNumberOfStates();
		int lenExample = input.size();

		// Best partial-path weights.
		double[][] delta = new double[lenExample][numberOfStates];
		// Best partial-path backward table.
		int[][] psi = new int[lenExample][numberOfStates];

		// Weights for the first token.
		int curState = partiallyLabeledOutput.getLabel(0);
		if (curState == nonAnnotatedStateCode) {
			for (int state = 0; state < numberOfStates; ++state) {
				delta[0][state] = getTokenEmissionWeightWithLoss(hmm, input, 0,
						state) + hmm.getInitialStateParameter(state);
			}
		} else {
			// Do not need to calculate anything. The next token will always
			// choose the labeled state as previous state despite delta values
			// (see partialViterbi method).
		}

		// Apply each step of the Viterbi algorithm.
		for (int tkn = 1; tkn < lenExample; ++tkn) {
			int prevState = curState;
			curState = partiallyLabeledOutput.getLabel(tkn);
			if (curState == nonAnnotatedStateCode) {
				// If the current token is non-annotated, we need to calculate
				// the best previous state and corresponding weight for each
				// possible state.
				for (int state = 0; state < numberOfStates; ++state)
					partialViterbi(hmm, prevState, delta, psi, input, tkn,
							state, defaultState);
			} else {
				// If the current token is annotated, we already know its state
				// and therefore only need to calculate the best previous state
				// to this annotated state.
				partialViterbi(hmm, prevState, delta, psi, input, tkn,
						curState, defaultState);
			}
		}

		// Find the best state for the last token.
		int lastState = partiallyLabeledOutput.getLabel(lenExample - 1);
		if (lastState == nonAnnotatedStateCode) {

			// The default state is always the fisrt option.
			int bestState = defaultState;
			double bestWeight = delta[lenExample - 1][defaultState];

			// Find the best last state.
			for (int state = 0; state < numberOfStates; ++state) {
				double weight = delta[lenExample - 1][state];
				if (weight > bestWeight) {
					bestWeight = weight;
					bestState = state;
				}
			}

			lastState = bestState;

		}

		// Reconstruct the best path from the best final state, annotating
		// the output sequence.
		backwardTag(predictedOutput, psi, lastState);
	}

	/**
	 * Find the best-weighted previous state for the given (current) token and
	 * state. If the previous token is annotated, the decision is
	 * straightforward.
	 * 
	 * @param hmm
	 *            the model
	 * @param previousState
	 *            the previous annotated state (or the special non-annotated
	 *            state).
	 * @param delta
	 *            the weight matrix
	 * @param psi
	 *            the backward path matrix
	 * @param input
	 *            the input sequence
	 * @param token
	 *            the token to be considered
	 * @param toState
	 *            the state to be considered
	 * @param defaultState
	 *            the default state that is chosen if every previous state
	 *            weights the same
	 */
	protected void partialViterbi(Hmm hmm, int previousState, double[][] delta,
			int[][] psi, SequenceInput input, int token, int toState,
			int defaultState) {
		if (previousState == nonAnnotatedStateCode) {
			// If the previous token is non-annotated, we must choose the
			// previous best state using the original procedure.
			viterbi(hmm, delta, psi, input, token, toState, defaultState);
		} else {
			// If the previous token is annotated, we choose the annotated state
			// as the best previous state.
			psi[token][toState] = previousState;
			delta[token][toState] = delta[token - 1][previousState]
					+ hmm.getTransitionParameter(previousState, toState)
					+ getTokenEmissionWeightWithLoss(hmm, input, token, toState);
		}
	}

	/**
	 * Sum the emission parameters of all features in the given token for the
	 * given state and, additionally, consider the loss function.
	 * 
	 * @param hmm
	 * @param input
	 * @param token
	 * @param state
	 * @return
	 */
	protected double getTokenEmissionWeightWithLoss(Hmm hmm,
			SequenceInput input, int token, int state) {
		double w = hmm.getTokenEmissionWeight(input, token, state);
		if (lossReferenceOutput != null
				&& lossReferenceOutput.getLabel(token) != state)
			w += lossWeight;
		return w;
	}

	/**
	 * Follow the given psi map from the last token (starting at the
	 * <code>bestFinalState</code>) to the first token of the sequence, tagging
	 * the given output sequence.
	 * 
	 * @param output
	 *            the output sequence to be tagged.
	 * @param psi
	 *            the psi map (backward map with the best previous state for
	 *            each token).
	 * @param bestFinalState
	 *            the best state for the last token of the sequence (this is the
	 *            start point).
	 */
	protected void backwardTag(SequenceOutput output, int[][] psi,
			int bestFinalState) {
		int len = output.size();
		for (int token = len - 1; token >= 0; --token) {
			output.setLabel(token, bestFinalState);
			bestFinalState = psi[token][bestFinalState];
		}
	}

}
