package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.io.PrintStream;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

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
	 * Return the total number of symbols (features) used in this model.
	 * 
	 * @return
	 */
	public abstract int getNumberOfSymbols();

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
	 * Set the value (weight) of the initial parameter associated with the given
	 * state.
	 * 
	 * @param state
	 * @param value
	 */
	public abstract void setInitialStateParameter(int state, double value);

	/**
	 * Set the value (weight) of the transition parameter associated with the
	 * given pair of states.
	 * 
	 * @param fromState
	 * @param toState
	 * @param value
	 */
	public abstract void setTransitionParameter(int fromState, int toState,
			double value);

	/**
	 * Set the value (weight) of the emission parameter associated with the
	 * given state-symbol pair.
	 * 
	 * @param state
	 * @param symbol
	 * @param value
	 */
	public abstract void setEmissionParameter(int state, int symbol,
			double value);

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
	 * Update the specified transition (fromToken, toToken) feature using the
	 * given learning rate.
	 * 
	 * @param fromToken
	 * @param toToken
	 * @param learningRate
	 */
	protected abstract void updateTransitionParameter(int fromToken,
			int toToken, double learningRate);

	/**
	 * Update the model features, corresponding to a given state, that are
	 * present in a token of the given input sequence. The given learning rate
	 * is used as a multiplier for each update.
	 * 
	 * @param input
	 * @param token
	 * @param state
	 * @param learningRate
	 */
	protected abstract void updateEmissionParameters(SequenceInput input,
			int token, int state, double learningRate);

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
		double accum = 0d;
		int numFtrs = input.getNumberOfInputFeatures(token);
		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr) {
			int ftr = input.getFeature(token, idxFtr);
			double weight = input.getFeatureWeight(token, idxFtr);
			accum += getEmissionParameter(state, ftr) * weight;
		}
		return accum;
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

		if (input.size() <= 0)
			return loss;

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

	public void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
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
