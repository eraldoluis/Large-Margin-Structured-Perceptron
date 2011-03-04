package br.pucrio.inf.learn.structlearning.application.sequence;

import java.io.PrintStream;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Model;

/**
 * Abstract class that represents a gereric HMM including the inference
 * algorithm (Viterbi) and the update procedure. The derived concrete classes
 * must represent the parameters internally and implement the methods to access
 * them (get's and update's).
 * 
 * @author eraldof
 * 
 */
public abstract class Hmm implements Model {

	/**
	 * Return the number of possible states (labels) of this model.
	 * 
	 * @return
	 */
	public abstract int getNumberOfStates();

	/**
	 * Return the weight associated with the given initial state.
	 * 
	 * @param state
	 * @return
	 */
	public abstract double getInitialStateParameter(int state);

	/**
	 * Return the weight associated with the transition from the two given
	 * states.
	 * 
	 * @param fromState
	 *            the origin state.
	 * @param toState
	 *            the end state.
	 * @return
	 */
	public abstract double getTransitionParameter(int fromState, int toState);

	/**
	 * Return the weight associated with the emission of the given symbol from
	 * the given state.
	 * 
	 * @param state
	 * @param symbol
	 * @return
	 */
	public abstract double getEmissionParameter(int state, int symbol);

	/**
	 * Add the given value to the initial-state parameter of the mobel.
	 * 
	 * @param state
	 * @param value
	 */
	protected abstract void updateInitialStateParameter(int state, double value);

	/**
	 * Add the given value to every feature at the token of the input sequence.
	 * 
	 * @param input
	 * @param token
	 * @param state
	 * @param value
	 */
	protected abstract void updateEmissionParameters(SequenceInput input,
			int token, int state, double value);

	/**
	 * Add the given value to the transition parameter.
	 * 
	 * @param fromToken
	 * @param toToken
	 * @param value
	 */
	protected abstract void updateTransitionParameter(int fromToken,
			int toToken, double value);

	/**
	 * The sub-classes must implement this to ease some use cases (e.g.,
	 * evaluating itermediate models during the execution of a training
	 * algorithm).
	 */
	public abstract Object clone() throws CloneNotSupportedException;

	/**
	 * Return the sum of the emission weights associated with the features in
	 * the token <code>token</code> of the sequence <code>input</code>.
	 * 
	 * @param input
	 * @param token
	 * @param state
	 * @return
	 */
	public double getTokenEmissionWeight(SequenceInput input, int token,
			int state) {
		double weight = 0d;
		for (int ftr : input.getFeatures(token))
			weight += getEmissionParameter(state, ftr);
		return weight;
	}

	/**
	 * Update the parameters of the features that differ from the two given
	 * output sequences and that are present in the given input sequence.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 * @return the loss between the correct and the predicted output.
	 */
	public double update(SequenceInput input, SequenceOutput outputCorrect,
			SequenceOutput outputPredicted, double learningRate) {

		double loss = 0d;

		// First token.
		int labelCorrect = outputCorrect.getLabel(0);
		int labelPredicted = outputPredicted.getLabel(0);
		if (labelCorrect != labelPredicted) {
			// Initial state parameters.
			updateInitialStateParameter(labelCorrect, learningRate);
			updateInitialStateParameter(labelPredicted, -learningRate);
			// Emission parameters.
			updateEmissionParameters(input, 0, labelCorrect, learningRate);
			updateEmissionParameters(input, 0, labelPredicted, -learningRate);
			// Update loss (per-token).
			loss += 1;
		}

		int prevLabelCorrect = labelCorrect;
		int prevLabelPredicted = labelPredicted;
		for (int tkn = 1; tkn < input.size(); ++tkn) {
			labelCorrect = outputCorrect.getLabel(tkn);
			labelPredicted = outputPredicted.getLabel(tkn);
			if (labelCorrect != labelPredicted) {
				// Emission parameters.
				updateEmissionParameters(input, tkn, labelCorrect, learningRate);
				updateEmissionParameters(input, tkn, labelPredicted,
						-learningRate);
				// Transition parameters.
				updateTransitionParameter(prevLabelCorrect, labelCorrect,
						learningRate);
				updateTransitionParameter(prevLabelPredicted, labelPredicted,
						-learningRate);
				// Update loss (per-token).
				loss += 1;
			} else if (prevLabelCorrect != prevLabelPredicted) {
				// Transition parameters.
				updateTransitionParameter(prevLabelCorrect, labelCorrect,
						learningRate);
				updateTransitionParameter(prevLabelPredicted, labelPredicted,
						-learningRate);
			}

			prevLabelCorrect = labelCorrect;
			prevLabelPredicted = labelPredicted;
		}

		return loss;
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((SequenceInput) input, (SequenceOutput) outputCorrect,
				(SequenceOutput) outputPredicted, learningRate);
	}

	public void save(PrintStream ps, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {
		ps.println("# initial state");
		for (int state = 0; state < getNumberOfStates(); ++state)
			ps.println(stateEncoding.getValueByCode(state) + "\t"
					+ getInitialStateParameter(state));
		ps.println("# transitions");
		for (int fromState = 0; fromState < getNumberOfStates(); ++fromState)
			for (int toState = 0; toState < getNumberOfStates(); ++toState)
				ps.println(stateEncoding.getValueByCode(fromState) + " "
						+ stateEncoding.getValueByCode(toState) + "\t"
						+ getTransitionParameter(fromState, toState));
		ps.println("# emissions");
		for (int state = 0; state < getNumberOfStates(); ++state)
			for (int symbol = 0; symbol < featureEncoding.size(); ++symbol) {
				ps.println(stateEncoding.getValueByCode(state) + " "
						+ featureEncoding.getValueByCode(symbol) + "\t"
						+ getEmissionParameter(state, symbol));
			}
	}

}
