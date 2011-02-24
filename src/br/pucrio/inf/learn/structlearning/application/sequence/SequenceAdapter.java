package br.pucrio.inf.learn.structlearning.application.sequence;

import java.util.List;
import java.util.Set;

import br.pucrio.inf.learn.structlearning.application.sequence.data.DetachedSequenceOutput;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.FeatureVector;
import br.pucrio.inf.learn.structlearning.task.TaskAdapter;

public class SequenceAdapter implements TaskAdapter {

	private int defaultState;

	private Set<Integer> labelSet;

	private List<SequenceFeatureTemplate> templates;

	@Override
	public FeatureVector extractFeatures(ExampleInput input,
			ExampleOutput output) {
		SequenceInput hmmInput = (SequenceInput) input;
		SequenceOutput hmmOutput = (SequenceOutput) output;

		FeatureVector features = new FeatureVector();
		for (int token = 0; token < hmmInput.size(); ++token)
			for (SequenceFeatureTemplate template : templates)
				features.increment(
						template.instance(hmmInput, hmmOutput, token), 1d);
		return features;
	}

	@Override
	public ExampleOutput inference(FeatureVector weight, ExampleInput input) {
		SequenceInput sInput = (SequenceInput) input;
		DetachedSequenceOutput output = new DetachedSequenceOutput(
				sInput.size());
		viterbi(weight, (SequenceInput) input, output);
		return output;
	}

	public void viterbi(FeatureVector weight, SequenceInput input,
			SequenceOutput output) {
		// Example length;
		int numberOfLabels = labelSet.size();
		int lenExample = input.size();

		double[][] delta = new double[lenExample][numberOfLabels];
		int[][] psi = new int[lenExample][numberOfLabels];

		// The log probabilities for the first token.
		boolean impossibleSymbol = true;
		for (int state = 0; state < numberOfLabels; ++state) {
			double emissionWeight = getEmissionParameter(
					input.getFeatureValue(0, observationFeature), state);
			psi[0][state] = -1;
			delta[0][state] = emissionWeight + getInitialStateParameter(state);

			if (delta[0][state] > Double.NEGATIVE_INFINITY)
				impossibleSymbol = false;
		}

		// Avoid impossible symbols (never seen) to degenerate the whole
		// prediction procedure.
		if (impossibleSymbol)
			delta[0][defaultState] = 0d;

		// Apply each step of the Viterb's algorithm.
		for (int tkn = 1; tkn < lenExample; ++tkn) {
			impossibleSymbol = true;
			for (int state = 0; state < numberOfLabels; ++state) {
				viterbi(delta, psi, input, tkn, state);

				if (delta[tkn][state] > Double.NEGATIVE_INFINITY)
					impossibleSymbol = false;
			}

			// Avoid impossible symbols (never seen) to degenerate the whole
			// prediction procedure.
			if (impossibleSymbol)
				delta[tkn][defaultState] = 0d;
		}

		// The default state is always the fisrt option.
		int bestState = defaultState;
		double maxLogProb = delta[lenExample - 1][defaultState];
		if (useFinalProbabilities)
			maxLogProb += getFinalStateParameter(bestState);

		// Find the best last state.
		for (int state = 0; state < numberOfLabels; ++state) {
			double logProb = delta[lenExample - 1][state];
			if (useFinalProbabilities)
				logProb += getFinalStateParameter(state);

			if (logProb > maxLogProb) {
				maxLogProb = logProb;
				bestState = state;
			}
		}

		// Reconstruct the best path from the best final state, and tag the
		// input.
		tagExample(input, psi, bestState);
	}

	protected void viterbi(double[][] delta, int[][] psi, SequenceInput input,
			int token, int state) {

		// Choose the best previous state.
		int maxState = defaultState;
		double maxLogProb = delta[token - 1][defaultState]
				+ getTransitionParameter(defaultState, state);
		for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
			double logProb = delta[token - 1][stateFrom]
					+ getTransitionParameter(stateFrom, state);
			if (logProb > maxLogProb) {
				maxLogProb = logProb;
				maxState = stateFrom;
			}
		}

		double emissionLogProb = getEmissionParameter(
				example.getFeatureValue(token, observationFeature), state);

		psi[token][state] = maxState;
		delta[token][state] = maxLogProb + emissionLogProb;
	}

	protected void tagExample(DatasetExample example, int[][] psi,
			int bestFinalState) {
		// Example length.
		int len = psi.length;

		example.setFeatureValue(len - 1, stateFeature,
				stateToFeature[bestFinalState]);
		for (int token = len - 1; token > 0; --token) {
			int stateFeatureVal = stateToFeature[psi[token][bestFinalState]];
			example.setFeatureValue(token - 1, stateFeature, stateFeatureVal);
			bestFinalState = psi[token][bestFinalState];
		}
	}

}
