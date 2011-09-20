package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Implement 2nd order Viterbi-based inference algorithms for sequence
 * structures.
 * 
 * @author eraldof
 * 
 */
public class Viterbi2ndOrderInference implements Inference {

	/**
	 * Default state to be chosen when all states weight the same.
	 */
	private int defaultState;

	/**
	 * Weight of the loss function in the objective function for annotated
	 * elements.
	 */
	private double lossAnnotatedWeight;

	/**
	 * Weight of the loss function in the objective function for NON annotated
	 * elements.
	 */
	private double lossNonAnnotatedWeight;

	/**
	 * This is the correct (or loss reference) output sequence corresponding to
	 * the current input sequence. This sequence is used to calculate the loss
	 * function.
	 */
	private SequenceOutput lossReferenceOutput;

	/**
	 * Output structure used to determine whether an element is annotated or
	 * not.
	 */
	private SequenceOutput lossPartiallyAnnotatedOutput;

	/**
	 * Create a Viterbi inference algorithm using the given state as the default
	 * option for every token.
	 * 
	 * @param defaultState
	 */
	public Viterbi2ndOrderInference(int defaultState) {
		this.defaultState = defaultState;
		this.lossAnnotatedWeight = 0d;
		this.lossNonAnnotatedWeight = 0d;
		this.lossReferenceOutput = null;
		this.lossPartiallyAnnotatedOutput = null;
	}

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		tag((Hmm2ndOrder) model, (SequenceInput) input, (SequenceOutput) output);
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
			ExampleOutput referenceOutput, ExampleOutput inferedOutput,
			double lossWeight) {
		// Save the current configuration.
		double previousLossWeight = this.lossAnnotatedWeight;
		SequenceOutput previousLossReferenceOutput = this.lossReferenceOutput;

		// Configure the loss-augmented necessary properties.
		this.lossAnnotatedWeight = lossWeight;
		this.lossReferenceOutput = (SequenceOutput) referenceOutput;

		// Call the ordinary inference algorithm.
		tag((Hmm2ndOrder) model, (SequenceInput) input,
				(SequenceOutput) inferedOutput);

		// Restore the previous configuration.
		this.lossAnnotatedWeight = previousLossWeight;
		this.lossReferenceOutput = previousLossReferenceOutput;
	}

	@Override
	public void lossAugmentedPartialInference(Model model, ExampleInput input,
			ExampleOutput lossPartiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput inferedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

	/**
	 * Tag the output with the best label sequence for the given input and HMM.
	 * 
	 * @param hmm
	 * @param input
	 * @param output
	 */
	public void tag(Hmm2ndOrder hmm, SequenceInput input, SequenceOutput output) {
		// Example length.
		int numStates = hmm.getNumberOfStates();
		int lenExample = input.size();

		if (lenExample <= 0)
			return;

		if (lenExample == 1) {
			// Examples of length 1 are special cases.
			int maxState = defaultState;
			double maxWeight = getLossAugmentedTokenEmissionWeight(hmm, input,
					0, maxState)
					+ hmm.getTransitionParameter(hmm.getNullState(),
							hmm.getNullState(), maxState);

			for (int state = 0; state < numStates; ++state) {
				// Weight for the first token at state 'state'.
				double weight = getLossAugmentedTokenEmissionWeight(hmm, input,
						0, state)
						+ hmm.getTransitionParameter(hmm.getNullState(),
								hmm.getNullState(), state);
				if (weight > maxWeight) {
					maxWeight = weight;
					maxState = state;
				}
			}

			output.setLabel(0, maxState);
			return;
		}

		// Best partial-path weights.
		double[][][] delta = new double[lenExample][numStates][numStates];
		// Best partial-path backward table.
		int[][][] psi = new int[lenExample][numStates][numStates];

		// Weights for the first token.
		for (int state = 0; state < numStates; ++state) {
			// Weight for the first token at state 'state'.
			delta[0][0][state] = getLossAugmentedTokenEmissionWeight(hmm,
					input, 0, state)
					+ hmm.getTransitionParameter(hmm.getNullState(),
							hmm.getNullState(), state);
		}

		// Weights for the second token.
		for (int state = 0; state < numStates; ++state) {
			// Constant emission weight.
			double emissionWeight = getLossAugmentedTokenEmissionWeight(hmm,
					input, 1, state);
			for (int prevState = 0; prevState < numStates; ++prevState) {
				/*
				 * Weight for the second token at state 'state' going through
				 * previous state 'prevState'.
				 */
				delta[1][prevState][state] = delta[0][0][prevState]
						+ hmm.getTransitionParameter(hmm.getNullState(),
								prevState, state) + emissionWeight;
			}
		}

		// Apply each step of the Viterbi algorithm from the third token on.
		for (int tkn = 2; tkn < lenExample; ++tkn)
			for (int state = 0; state < numStates; ++state)
				viterbi(hmm, delta, psi, input, tkn, state, defaultState);

		// The default state is always the first option.
		int bestLastState = defaultState;
		int bestLastButOneState = defaultState;
		double bestWeight = delta[lenExample - 1][bestLastButOneState][bestLastState];

		// Find the best last and last but one states.
		for (int lastState = 0; lastState < numStates; ++lastState) {
			for (int lastButOneState = 0; lastButOneState < numStates; ++lastButOneState) {
				double weight = delta[lenExample - 1][lastButOneState][lastState];
				if (weight > bestWeight) {
					bestWeight = weight;
					bestLastState = lastState;
					bestLastButOneState = lastButOneState;
				}
			}
		}

		// Reconstruct the best path from the best final state, and tag the
		// input.
		backwardTag(output, psi, bestLastButOneState, bestLastState);
	}

	/**
	 * Calculate the best previous state (fromState) to the given end-state
	 * <code>toState</code> for the given token <code>token</code>.
	 * 
	 * @param hmm
	 *            the HMM to be used
	 * @param delta
	 *            the best accumulated weights until the previous token
	 * @param psi
	 *            used to store the best option
	 * @param input
	 *            the input structure
	 * @param token
	 *            the token be considered
	 * @param finalState
	 *            the final state to be considered
	 * @param defaultState
	 *            the default state
	 */
	protected void viterbi(Hmm2ndOrder hmm, double[][][] delta, int[][][] psi,
			SequenceInput input, int token, int finalState, int defaultState) {
		// Number of states.
		int numStates = hmm.getNumberOfStates();

		// Fixed emission weight for the given token at the given state.
		double emissionWeight = getLossAugmentedTokenEmissionWeight(hmm, input,
				token, finalState);

		// For all possible previous states.
		for (int prevState = 0; prevState < numStates; ++prevState) {

			// Choose the best state before the previous state 'prevState'.
			int maxPrevPrevState = defaultState;
			double maxPrevPrevWeight = delta[token - 1][maxPrevPrevState][prevState]
					+ hmm.getTransitionParameter(maxPrevPrevState, prevState,
							finalState);

			for (int prevPrevState = 0; prevPrevState < numStates; ++prevPrevState) {
				double weight = delta[token - 1][prevPrevState][prevState]
						+ hmm.getTransitionParameter(prevPrevState, prevState,
								finalState);
				if (weight > maxPrevPrevWeight) {
					maxPrevPrevWeight = weight;
					maxPrevPrevState = prevPrevState;
				}
			}

			/*
			 * Max state before previous state (maxPrevPrevState) going through
			 * previous state prevState.
			 */
			psi[token][prevState][finalState] = maxPrevPrevState;
			delta[token][prevState][finalState] = maxPrevPrevWeight
					+ emissionWeight;
		}
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
		throw new NotImplementedException();
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
		throw new NotImplementedException();
	}

	/**
	 * Return the sum of the emission parameters of all features in the given
	 * token and state and, additionally, augment this value with the loss
	 * function if the user specified so.
	 * 
	 * @param hmm
	 * @param input
	 * @param token
	 * @param state
	 * @return
	 */
	protected double getLossAugmentedTokenEmissionWeight(Hmm2ndOrder hmm,
			SequenceInput input, int token, int state) {

		// The ordinary emission weight for the current token.
		double w = hmm.getTokenEmissionWeight(input, token, state);

		// Augment the objective function value with a possible loss.
		if (lossReferenceOutput != null
				&& lossReferenceOutput.getLabel(token) != state) {
			// If the user provided a loss-reference output structure and the
			// reference token label is different from the predicted one, then
			// count the loss value for this token.
			if (lossPartiallyAnnotatedOutput == null
					|| lossPartiallyAnnotatedOutput.getLabel(token) == lossReferenceOutput
							.getLabel(token))
				// If the user did not provide a partially-labeled output
				// structure, or if he/she did but the token is annotated.
				w += lossAnnotatedWeight;
			else
				// If the user provided a partially-labeled output structure and
				// the token is NON-annotated.
				w += lossNonAnnotatedWeight;
		}

		return w;

	}

	/**
	 * Follow psi table to fill the given output.
	 * 
	 * @param output
	 *            output sequence to be filled.
	 * @param psi
	 *            backward table of the best path.
	 * @param prevState
	 *            the best last but one state.
	 * @param state
	 *            the best last state.
	 */
	protected void backwardTag(SequenceOutput output, int[][][] psi,
			int prevState, int state) {
		int len = output.size();
		for (int token = len - 1; token >= 0; --token) {
			output.setLabel(token, state);
			int aux = prevState;
			prevState = psi[token][prevState][state];
			state = aux;
		}
	}

}
