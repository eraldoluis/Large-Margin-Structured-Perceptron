package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Abstract class that represents a 2nd order HMM including the inference
 * algorithm (Viterbi) and the update procedure. The derived concrete classes
 * must represent the parameters internally and implement the methods to access
 * them (get's and update's).
 * 
 * @author eraldof
 * 
 */
public abstract class Hmm2ndOrder implements Model {

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
	 * The null state is used to represent the previous (constant) state before
	 * the first token and the one before it.
	 * 
	 * @return
	 */
	public abstract int getNullState();

	/**
	 * Return the weight associated with the transition from two previous states
	 * to a third one.
	 * 
	 * @param state1
	 *            the state before the previous state.
	 * @param state2
	 *            the previous state.
	 * @param state3
	 *            the current state.
	 * @return
	 */
	public abstract double getTransitionParameter(int state1, int state2,
			int state3);

	/**
	 * Set the value (weight) of the transition parameter associated with the
	 * given triple of states.
	 * 
	 * @param state1
	 *            the state before the previous one.
	 * @param state2
	 *            the previous state.
	 * @param state3
	 *            the current state.
	 * @param value
	 */
	public abstract void setTransitionParameter(int state1, int state2,
			int state3, double value);

	/**
	 * Increment the transition parameter associated with the given states by
	 * the given increment value.
	 * 
	 * @param state1
	 *            the state before the previous one.
	 * @param state2
	 *            the previous state.
	 * @param state3
	 *            the current state.
	 * @param value
	 *            the increment value.
	 */
	protected abstract void updateTransitionParameter(int state1, int state2,
			int state3, double value);

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
	 * evaluating intermediate models during the execution of a training
	 * algorithm).
	 */
	public abstract Hmm2ndOrder clone() throws CloneNotSupportedException;

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

		if (input.size() <= 0)
			return 0d;

		/*
		 * Accumulated loss (number of misclassified tokens) within the given
		 * example.
		 */
		double loss = 0d;

		// Before previous token labels;
		int prevPrevLabelCorrect = getNullState();
		int prevprevLabelPredicted = getNullState();

		// Previous token labels.
		int prevLabelCorrect = getNullState();
		int prevLabelPredicted = getNullState();

		for (int tkn = 0; tkn < input.size(); ++tkn) {
			// Current token labels (correct and predicted).
			int labelCorrect = outputCorrect.getLabel(tkn);
			int labelPredicted = outputPredicted.getLabel(tkn);

			// Current classification is wrong?
			if (labelCorrect != labelPredicted) {
				// Increment correct emission parameters.
				updateEmissionParameters(input, tkn, labelCorrect, learningRate);
				// Decrement incorrect emission parameters.
				updateEmissionParameters(input, tkn, labelPredicted,
						-learningRate);
				// Increment correct transition parameters.
				updateTransitionParameter(prevPrevLabelCorrect,
						prevLabelCorrect, labelCorrect, learningRate);
				// Decrement incorrect transition parameters.
				updateTransitionParameter(prevprevLabelPredicted,
						prevLabelPredicted, labelPredicted, -learningRate);
				// Update loss (per-token misclassification).
				loss += 1;
			} else if (prevLabelCorrect != prevLabelPredicted
					|| prevPrevLabelCorrect != prevprevLabelPredicted) {
				// Increment correct transition parameters.
				updateTransitionParameter(prevPrevLabelCorrect,
						prevLabelCorrect, labelCorrect, learningRate);
				// Decrement incorrect transition parameters.
				updateTransitionParameter(prevprevLabelPredicted,
						prevLabelPredicted, labelPredicted, -learningRate);
			}

			// Advance to next token.
			prevPrevLabelCorrect = prevLabelCorrect;
			prevprevLabelPredicted = prevLabelPredicted;
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

	@Override
	public void save(String fileName, Dataset dataset) {
		throw new NotImplementedException();
	}

}
