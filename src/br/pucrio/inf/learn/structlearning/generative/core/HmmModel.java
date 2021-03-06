package br.pucrio.inf.learn.structlearning.generative.core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.generative.data.FeatureValueEncoding;
import br.pucrio.inf.learn.structlearning.generative.data.HmmModelStrings;
import br.pucrio.inf.learn.util.RandomGenerator;

/**
 * Store a hidden Markov model parameters and provide some services over this
 * model. Implement some smoothing techniques and a tagging algorithm (Viterbi).
 * 
 * @author eraldof
 * 
 */
public class HmmModel {

	static Log LOG = LogFactory.getLog(HmmModel.class);

	/**
	 * For non-seen observations, use the state with this label.
	 */
	private String defaultStateLabel = "0";

	/**
	 * For non-seen observations, use this state.
	 */
	private int defaultState = -1;

	/**
	 * Control the mapping from string feature values to integer values and
	 * vice-versa. This model needs to know about this mapping only to serialize
	 * and unserialize.
	 */
	private FeatureValueEncoding featureValueEncoding;

	/**
	 * Number of states in this HMM.
	 */
	protected int numStates;

	/**
	 * Map from feature values to state codes (that varies from 0 to NUM_STATES
	 * - 1).
	 * 
	 * This is needed because the states are given by the values of some feature
	 * in the training dataset. However, these feature values are unbounded (may
	 * vary too much). So, here, we map each feature value (that is a state) to
	 * an integer value from 0 to (numStates - 1). So, the state values has two
	 * levels of encoding. First, the string representation of a state is
	 * converted to an integer code using the <code>FeatureValueEncoding</code>
	 * associated with the training dataset. Later, in this class, the integer
	 * feature values that represent states are converted to a bounded integer
	 * value (from 0 to numStates - 1) to ease its use as array index.
	 */
	protected HashMap<Integer, Integer> featureToState;

	/**
	 * Map from state codes (that varies from 0 to NUM_STATES - 1) to feature
	 * values.
	 * 
	 * This is needed because the states are given by the values of some feature
	 * in the training dataset. However, these feature values are unbounded (may
	 * vary too much). So, here, we map each feature value (that is a state) to
	 * an integer value from 0 to (numStates - 1). So, the state values has two
	 * levels of encoding. First, the string representation of a state is
	 * converted to an integer code using the <code>FeatureValueEncoding</code>
	 * associated with the training dataset. Later, in this class, the integer
	 * feature values that represent states are converted to a bounded integer
	 * value (from 0 to numStates - 1) to ease its use as array index.
	 */
	protected int[] stateToFeature;

	/**
	 * Initial state probabilities.
	 * 
	 * The value <code>probInitialState[i]</code> is the probability of starting
	 * at state <code>i</code>. This means that we have a special initial state.
	 * One may not use this special state and, instead, use the default 0 state.
	 */
	protected double[] probInitialState;
	protected double[] lnProbInitialState;

	/**
	 * Final state probabilities.
	 * 
	 * The value <code>probFinalState[i]</code> is the probability of ending at
	 * state <code>i</code>. This means that we have a special final state. One
	 * may not use this special state and, instead, use the default 0 state.
	 */
	protected double[] probFinalState;
	protected double[] lnProbFinalState;

	/**
	 * Transition probabilities.
	 * 
	 * The value <code>probTransition[i][j]</code> is the probability of going
	 * to state <code>j</code> when you are in state <code>i</code>.
	 * 
	 */
	protected double[][] probTransition;
	protected double[][] lnProbTransition;

	/**
	 * Emission probabilities.
	 * 
	 * The value <code>probEmission[i].get(f)</code> is the probability to emit
	 * the observation <code>f</code> when in state <code>i</code>.
	 */
	protected Vector<HashMap<Integer, Double>> probEmission;
	protected Vector<HashMap<Integer, Double>> lnProbEmission;

	/**
	 * The index of the feature, within the dataset passed to the
	 * <code>tag</code> method, that contains the observation values.
	 * 
	 * This is temporary property used by methods called from the
	 * <code>tag</code> method.
	 */
	protected int observationFeature;

	/**
	 * The index of the feature, within the dataset passed to the
	 * <code>tag</code> method, that contains the state values.
	 * 
	 * This is temporary property used by methods called from the
	 * <code>tag</code> method.
	 */
	protected int stateFeature;

	/**
	 * Indicate when to use or not the final probilities parameters to tag
	 * sequences.
	 */
	protected boolean useFinalProbabilities;

	/**
	 * Smoothing techniques.
	 * 
	 * @author eraldof
	 * 
	 */
	public enum Smoothing {
		/**
		 * No smoothing.
		 */
		NONE,

		/**
		 * Laplace smoothing.
		 */
		LAPLACE,

		/**
		 * Absolute-discount smoothing.
		 */
		ABSOLUTE_DISCOUNTING,
	}

	/**
	 * If true, use absolute discount smoothing.
	 */
	protected Smoothing smoothing;

	/**
	 * Indicate if the model has already been normalized or, otherwise, still
	 * contains the occurrences counters.
	 */
	protected boolean normalized;

	/**
	 * Create a new HMM model using the given <code>FeatureValueEncoding</code>.
	 * 
	 * An HMM model uses a <code>FeatureValueEncoding</code> only to serialize
	 * itself.
	 * 
	 * @param featureValueEncoding
	 * @param featureValuesThatAreStates
	 */
	public HmmModel(FeatureValueEncoding featureValueEncoding,
			Collection<Integer> featureValuesThatAreStates) {
		this.featureValueEncoding = featureValueEncoding;
		initStateFeatures(featureValuesThatAreStates);
		allocProbabilityParameters();
	}

	/**
	 * Create a new HMM model and load the parameters from the given file.
	 * 
	 * @param fileName
	 *            the filename from where to load the model parameters.
	 * @param featureValueEncoding
	 *            the feature-value encoding mapping to be used.
	 * 
	 * @throws IOException
	 *             if some problem occurs when loading the file.
	 */
	public HmmModel(String fileName, FeatureValueEncoding featureValueEncoding)
			throws IOException {
		this.featureValueEncoding = featureValueEncoding;
		load(fileName);
	}

	/**
	 * Create a new HMM model and load the parameters from the given file.
	 * 
	 * @param fileName
	 *            name of the file where the model is loaded from.
	 * 
	 * @throws IOException
	 *             if some problem occurs when opening or parsing the model
	 *             file.
	 */
	public HmmModel(String fileName) throws IOException {
		this.featureValueEncoding = new FeatureValueEncoding();
		load(fileName);
	}

	/**
	 * Create a new HMM by loading a model from a file generated by a Hadoop
	 * trainer.
	 * 
	 * @param fileName
	 * @param fromHadoop
	 * @throws IOException
	 */
	public HmmModel(String fileName, boolean fromHadoop) throws IOException {
		this.featureValueEncoding = new FeatureValueEncoding();
		if (fromHadoop)
			loadFromHadoopModel(fileName);
		else
			load(fileName);
	}

	public HmmModel(String fileName, boolean fromHadoop, HmmModel baseModel)
			throws IOException {

		this.numStates = baseModel.numStates;
		this.featureToState = baseModel.featureToState;
		this.stateToFeature = baseModel.stateToFeature;
		this.featureValueEncoding = baseModel.featureValueEncoding;

		if (fromHadoop)
			loadFromHadoopModel(fileName);
		else
			load(fileName);
	}

	private void initStateFeatures(
			Collection<Integer> featureValuesThatAreStates) {
		numStates = featureValuesThatAreStates.size();
		stateToFeature = new int[numStates];
		featureToState = new HashMap<Integer, Integer>();

		// Map feature values within the given encoding to state indexes (from 0
		// to numStates - 1).
		int st = 0;
		for (Integer ftrVal : featureValuesThatAreStates) {
			stateToFeature[st] = ftrVal;
			featureToState.put(ftrVal, st);
			++st;
		}
	}

	private void allocProbabilityParameters() {
		probInitialState = new double[numStates];
		probFinalState = new double[numStates];
		probTransition = new double[numStates][numStates];
		probEmission = new Vector<HashMap<Integer, Double>>(numStates);
		probEmission.setSize(numStates);
	}

	public boolean getUseFinalProbabilities() {
		return useFinalProbabilities;
	}

	public void setUseFinalProbabilities(boolean useFinalProbabilities) {
		this.useFinalProbabilities = useFinalProbabilities;
	}

	public String getDefaultStateLabel() {
		return defaultStateLabel;
	}

	public void setDefaultStateLabel(String defaultStateLabel) {
		this.defaultStateLabel = defaultStateLabel;
	}

	/**
	 * Tag the given dataset.
	 * 
	 * @param dataset
	 * @param stateFeature
	 * @param observationFeature
	 * @throws DatasetException
	 */
	public void tag(Corpus dataset, String observationFeatureLabel,
			String stateFeatureLabel) throws DatasetException {
		// Get the indexes of the observation and state features.
		this.observationFeature = dataset
				.getFeatureIndex(observationFeatureLabel);

		this.stateFeature = dataset.getFeatureIndex(stateFeatureLabel);
		if (this.stateFeature < 0)
			this.stateFeature = dataset.createNewFeature(stateFeatureLabel);

		// Get the default state.
		this.defaultState = getStateByString(defaultStateLabel);
		if (this.defaultState < 0)
			this.defaultState = 0;

		if (dataset.getFeatureValueEncoding() != featureValueEncoding)
			throw new DatasetException(
					"The dataset given for tagging has a different feature-value encoding."
							+ "You need to use the same.");

		for (int idxExample = 0; idxExample < dataset.getNumberOfExamples(); ++idxExample)
			viterbi(dataset.getExample(idxExample));
	}

	public void tag(Corpus dataset, int observationFeature, int stateFeature)
			throws DatasetException {

		if (dataset.getFeatureValueEncoding() != featureValueEncoding)
			throw new DatasetException(
					"The dataset given for tagging has a different feature-value encoding."
							+ "You need to use the same.");

		// Get the indexes of the observation and state features.
		this.observationFeature = observationFeature;

		this.stateFeature = stateFeature;
		if (this.stateFeature < 0)
			this.stateFeature = dataset.createNewFeature("label");

		// Get the default state.
		this.defaultState = getStateByString(defaultStateLabel);
		if (this.defaultState < 0)
			this.defaultState = 0;

		for (int idxExample = 0; idxExample < dataset.getNumberOfExamples(); ++idxExample)
			viterbi(dataset.getExample(idxExample));
	}

	/**
	 * Generate a bunch of examples in the given dataset by using this HMM
	 * model.
	 * 
	 * @param dataset
	 *            a dataset to store the generated examples.
	 * @param observationFeatureLabel
	 *            the feature label used as observation.
	 * @param stateFeatureLabel
	 *            the feature label used as state.
	 * @param numberOfExamples
	 *            the number of examples to be generated.
	 * @param length
	 *            the mean length (in tokens) of the examples.
	 * @param standardDeviation
	 *            the standard deviation of the examples length. If this value
	 *            is zero, all examples will have the same length. If it is
	 *            greater than zero then the example length will be choosen
	 *            using a normal distribution with the given mean and this
	 *            standard deviation.
	 * 
	 * @throws DatasetException
	 *             if the given feature labels do not exist or the
	 *             <code>FeatureValueEncoding</code> of this model and the given
	 *             dataset are not the same.
	 */
	public void generateExamples(Corpus dataset,
			String observationFeatureLabel, String stateFeatureLabel,
			int numberOfExamples, double lengthMean, double standardDeviation)
			throws DatasetException {
		// Check the encoding object.
		if (dataset.getFeatureValueEncoding() != featureValueEncoding)
			throw new DatasetException(
					"The given dataset must have the same feature-value encoding of this model.");

		// Features indexes.
		int obsFtr = dataset.getFeatureIndex(observationFeatureLabel);
		int staFtr = dataset.getFeatureIndex(stateFeatureLabel);

		// A base vector to use as "seed".
		Vector<Integer> baseToken = new Vector<Integer>(
				dataset.getNumberOfFeatures());
		baseToken.setSize(dataset.getNumberOfFeatures());
		for (int idx = 0; idx < baseToken.size(); ++idx)
			baseToken.set(idx,
					dataset.getFeatureValueEncoding().putString("-X-X-X-"));

		for (int i = 0; i < numberOfExamples; ++i) {
			// Probabilistically adjust the example length according to the
			// given.
			double length = lengthMean;
			if (standardDeviation > 0.0) {
				double var = 0.0;
				do {
					var = RandomGenerator.gen.nextGaussian()
							* standardDeviation;
				} while (Math.round(length + var) < 1.0);
				length += var;
			}

			// Generate the example and insert it in the dataset.
			generateExample(dataset, "" + i, baseToken, obsFtr, staFtr,
					(int) Math.round(length));
		}
	}

	/**
	 * Generate an example using this HMM model and insert it in the given
	 * dataset.
	 * 
	 * @param dataset
	 *            the dataset to store the generated example.
	 * @param id
	 *            the identification string of the example.
	 * @param baseToken
	 *            a template to generate the tokens of the example (since only
	 *            the observation and the state features are generated, the
	 *            other features must receive some default values in every
	 *            token).
	 * @param observationFeature
	 *            index of the observation feature.
	 * @param stateFeature
	 *            index of the state feature.
	 * @param length
	 *            length of the example.
	 */
	protected void generateExample(Corpus dataset, String id,
			Vector<Integer> baseToken, int observationFeature,
			int stateFeature, int length) {

		// The example.
		Vector<Vector<Integer>> example = new Vector<Vector<Integer>>(length);

		// Randomly choose the first state.
		int state = probChoose(probInitialState);

		// Fill the next states/observations.
		for (int idxTkn = 0; idxTkn < length; ++idxTkn) {
			// Randomly choose the emission given the current state.
			int emission = probChoose(probEmission.get(state));

			// Create a new token from the base token and add it to the example.
			@SuppressWarnings("unchecked")
			Vector<Integer> token = (Vector<Integer>) baseToken.clone();
			example.add(token);

			// Fill the observation and state features in the token.
			token.set(observationFeature, emission);
			token.set(stateFeature, stateToFeature[state]);

			// Randomly choose the next state.
			state = probChoose(probTransition[state]);
		}

		try {
			dataset.addExample(id, example);
		} catch (DatasetException e) {
		}
	}

	/**
	 * Add a Gaussian signal to this model parameters.
	 * 
	 * For each parameters, add a probability value sampled from a normal
	 * distribution with mean of zero and standard deviation equal to the given
	 * value.
	 * 
	 * @param standardDeviation
	 *            the standard deviation of the added Gaussian signal.
	 */
	public void addNormalNoise(double standardDeviation) {
		// Alias to the random number generator.
		Random r = RandomGenerator.gen;

		double normFactorInit = mean(probInitialState);
		double normFactorFinal = mean(probFinalState);

		for (int state = 0; state < numStates; ++state) {
			// Initial state probabilities.
			probInitialState[state] = Math.max(0.0, probInitialState[state]
					+ normFactorInit * r.nextGaussian() * standardDeviation);

			// Final state probabilities.
			probFinalState[state] = Math.max(0.0, probFinalState[state]
					+ normFactorFinal * r.nextGaussian() * standardDeviation);

			// Transition probabilities.
			double normFactorTrans = mean(probTransition[state]);
			for (int stateTo = 0; stateTo < numStates; ++stateTo)
				if (probTransition[state][stateTo] > 0.0)
					probTransition[state][stateTo] = Math.max(0.0,
							probTransition[state][stateTo] + normFactorTrans
									* r.nextGaussian() * standardDeviation);

			// Skip unseen states.
			Map<Integer, Double> emissionMap = probEmission.get(state);
			if (emissionMap == null)
				continue;

			/*
			 * Emission probabilities.
			 */

			// Set of symbols allowed to be added and replace the removed
			// symbols. At start, all symbols are allowed.
			HashSet<Integer> toBeAdded = new HashSet<Integer>(
					featureValueEncoding.getCollectionOfLabels());

			// List of symbols to be removed because their probabilities lie
			// below zero.
			LinkedList<Integer> toBeRemoved = new LinkedList<Integer>();

			double normFactorEmis = mean(emissionMap.values());
			for (Entry<Integer, Double> emission : emissionMap.entrySet()) {
				// Allow to add only new symbols.
				toBeAdded.remove(emission.getKey());

				Double prob = emission.getValue();
				prob += normFactorEmis * r.nextGaussian() * standardDeviation;
				if (prob > 0.0)
					emission.setValue(prob);
				else
					toBeRemoved.add(emission.getKey());
			}

			// Clean probabilities less than or equal to zero.
			for (Integer key : toBeRemoved)
				emissionMap.remove(key);

			// Add to the emission map of this state the same number of emission
			// symbols that are being removed. Add ramdom symbols that were not
			// present in this emission map.
			for (int added = 0; added < toBeRemoved.size(); ++added) {
				// Randomly choose an index in the toBeAdded set.
				int toAdd = r.nextInt() % toBeAdded.size();

				// Find the symbol at the chosen index.
				int idx = 0;
				Iterator<Integer> it = toBeAdded.iterator();
				while (idx < toAdd) {
					it.next();
					++idx;
				}
				toAdd = it.next();

				it.remove();

				// Randomly choose a probability to this emission (ensure a
				// greater than zero probability).
				double prob = Math.abs(normFactorEmis * r.nextGaussian()
						* standardDeviation);
				while (prob <= 0.0)
					prob = Math.abs(normFactorEmis * r.nextGaussian()
							* standardDeviation);
				emissionMap.put(toAdd, prob);
			}
		}

		// Normalize the probabilities and apply the log.
		normalizeProbabilities();
		applyLog();
	}

	/**
	 * Tag the given example using the Viterbi algorithm.
	 * 
	 * @param example
	 */
	protected void viterbi(DatasetExample example) {
		int lenExample = example.size();

		double[][] delta = new double[lenExample][numStates];
		int[][] psi = new int[lenExample][numStates];

		// The log probabilities for the first token.
		boolean impossibleSymbol = true;
		for (int state = 0; state < numStates; ++state) {
			double emissionWeight = getEmissionParameter(
					example.getFeatureValue(0, observationFeature), state);

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
			for (int state = 0; state < numStates; ++state) {
				viterbi(delta, psi, example, tkn, state);

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
		for (int state = 0; state < numStates; ++state) {
			double logProb = delta[lenExample - 1][state];
			if (useFinalProbabilities)
				logProb += getFinalStateParameter(state);

			if (logProb > maxLogProb) {
				maxLogProb = logProb;
				bestState = state;
			}
		}

		// Reconstruct the best path from the best final state, and tag the
		// example.
		tagExample(example, psi, bestState);
	}

	protected void viterbi(double[][] delta, int[][] psi,
			DatasetExample example, int token, int state) {

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

	/**
	 * Tag the given example using the given psi table.
	 * 
	 * @param example
	 * @param delta
	 * @param psi
	 * @param bestFinalState
	 */
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

	/**
	 * Probabilistically choose an index of the given array using the values in
	 * the array as probabilities.
	 * 
	 * @param probDistribution
	 *            values that are proportional to the probability of choose an
	 *            element.
	 * 
	 * @return an index within the given array.
	 */
	private int probChoose(double[] probDistribution) {
		int idx = 0;
		double accum = probDistribution[0];
		double val = RandomGenerator.gen.nextDouble();
		while (val >= accum && idx < probDistribution.length) {
			++idx;
			accum += probDistribution[idx];
		}

		return idx;
	}

	/**
	 * Probabilistically choose an item of the given map using the values in the
	 * map as probability weights.
	 * 
	 * @param weights
	 *            a map whose item values are proportional to the probability of
	 *            choose an element.
	 * 
	 * @return the key of the choosen item.
	 */
	private int probChoose(Map<Integer, Double> weights) {
		Entry<Integer, Double> choosen = null;
		double accum = 0.0;
		double val = RandomGenerator.gen.nextDouble();
		for (Entry<Integer, Double> current : weights.entrySet()) {
			choosen = current;
			accum += current.getValue();
			if (accum > val)
				break;
		}

		return choosen.getKey();
	}

	/**
	 * Return the parameter associated with the probability of ending at state
	 * <code>state</code>.
	 * 
	 * @param state
	 * 
	 * @return
	 */
	public double getFinalStateParameter(int state) {
		return lnProbFinalState[state];
	}

	/**
	 * Return the probability of ending a sequence of observations in the given
	 * state.
	 * 
	 * @param state
	 *            a state index.
	 * 
	 * @return the probability of ending a sequence of observations in the given
	 *         state.
	 */
	public double getFinalStateProbability(int state) {
		return probFinalState[state];
	}

	/**
	 * Return the parameter associated with the probability of starting at state
	 * <code>state</code>.
	 * 
	 * @param state
	 * 
	 * @return
	 */
	public double getInitialStateParameter(int state) {
		return lnProbInitialState[state];
	}

	/**
	 * Return the probability of starting in the given state.
	 * 
	 * @param state
	 *            a state index.
	 * 
	 * @return the probability of starting in the given state.
	 */
	public double getInitialStateProbability(int state) {
		return probInitialState[state];
	}

	/**
	 * Return the parameter associated with the transition form state
	 * <code>fromState</code> to state <code>toState</code>.
	 * 
	 * @param fromState
	 *            the origin state.
	 * @param toState
	 *            the destination state.
	 * 
	 * @return the parameter valued associated with the given transition.
	 */
	public double getTransitionParameter(int fromState, int toState) {
		return lnProbTransition[fromState][toState];
	}

	/**
	 * Return the probability of transition from state <code>fromState</code> to
	 * state <code>toState</code>.
	 * 
	 * @param fromState
	 *            a state index.
	 * @param toState
	 *            a state index.
	 * 
	 * @return the probability of transition.
	 */
	public double getTransitionProbability(int fromState, int toState) {
		return probTransition[fromState][toState];
	}

	/**
	 * Return the parameter associated with the probability of the emission of
	 * the word features at the given state. This is the log of the probability
	 * itself.
	 * 
	 * @param symbol
	 *            the emission feature value.
	 * @param state
	 *            the state/tag.
	 * 
	 * @return the parameter associated with the emission probability
	 *         (state,wordFeatures).
	 */
	public double getEmissionParameter(int symbol, int state) {
		HashMap<Integer, Double> lnEmissionMap = lnProbEmission.get(state);
		if (lnEmissionMap == null)
			return Double.NEGATIVE_INFINITY;

		Double lnProb = lnEmissionMap.get(symbol);
		if (lnProb == null)
			return Double.NEGATIVE_INFINITY;

		return lnProb;
	}

	/**
	 * Return the probability of emission of the given symbol in the given
	 * state.
	 * 
	 * @param symbol
	 *            a code for a symbol within the symbol dictionary.
	 * @param state
	 *            a state index.
	 * 
	 * @return the probability of emitting the symbol in the state.
	 */
	public double getEmissionProbability(int symbol, int state) {
		HashMap<Integer, Double> emissionMap = probEmission.get(state);
		if (emissionMap == null)
			return 0.0;

		Double prob = emissionMap.get(symbol);
		if (prob == null)
			return 0.0;

		return prob;
	}

	/**
	 * Increment by one unit the initial state counter (that is accumulate in
	 * the probability attribute).
	 * 
	 * The parameter <code>stateFtr</code> must be a feature value within the
	 * <code>FeatureValueEncoding</code> given in the constructor of this model.
	 * This value will be converted to a state index.
	 * 
	 * @param stateFtr
	 *            the feature value indicating a state.
	 */
	public void incProbInitialByFeature(int stateFtr) {
		int state = featureToState.get(stateFtr);
		++probInitialState[state];
	}

	public void incProbInitialByFeature(int stateFtr, double value) {
		int state = featureToState.get(stateFtr);
		probInitialState[state] += value;
	}

	public void setInitialStateProbability(int stateCode, double value) {
		probInitialState[stateCode] = value;
	}

	public void setInitialStateProbabilityByFeature(int stateFtr, double value) {
		int stateCode = featureToState.get(stateFtr);
		probInitialState[stateCode] = value;
	}

	/**
	 * Increment by one unit the final state counter (that is accumulate in the
	 * probability attribute).
	 * 
	 * The parameter <code>stateFtr</code> must be a feature value within the
	 * <code>FeatureValueEncoding</code> given in the constructor of this model.
	 * This value will be converted to a state index.
	 * 
	 * @param stateFtr
	 *            the feature value indicating a state.
	 */
	public void incProbFinalByFeature(int stateFtr) {
		int state = featureToState.get(stateFtr);
		++probFinalState[state];
	}

	public void incProbFinalByFeature(int stateFtr, double value) {
		int state = featureToState.get(stateFtr);
		probFinalState[state] += value;
	}

	/**
	 * Increment by one unit the transition counter.
	 * 
	 * The parameters <code>stateFromFtr</code> and <code>stateToFtr</code> must
	 * be a feature value within the <code>FeatureValueEncoding</code> given in
	 * the constructor of this model. This value will be converted to a state
	 * index.
	 * 
	 * @param stateFromFtr
	 *            the feature value indicating the source state.
	 * @param stateToFtr
	 *            the feature value indicating the target state.
	 */
	public void incProbTransitionByFeature(int stateFromFtr, int stateToFtr) {
		int stateFrom = featureToState.get(stateFromFtr);
		int stateTo = featureToState.get(stateToFtr);
		++probTransition[stateFrom][stateTo];
	}

	public void incProbTransitionByFeature(int stateFromFtr, int stateToFtr,
			double value) {
		int stateFrom = featureToState.get(stateFromFtr);
		int stateTo = featureToState.get(stateToFtr);
		probTransition[stateFrom][stateTo] += value;
	}

	public void setProbTransition(int stateFrom, int stateTo, double value) {
		probTransition[stateFrom][stateTo] = value;
	}

	public void setProbTransitionByFeature(int stateFromFtr, int stateToFtr,
			double value) {
		int stateFrom = featureToState.get(stateFromFtr);
		int stateTo = featureToState.get(stateToFtr);
		probTransition[stateFrom][stateTo] = value;
	}

	/**
	 * Increment by one unit the value
	 * <code>probEmission.get(state).get(emission)</code>.
	 * 
	 * This method is useful because these probabilities are stored in a Vector
	 * of HashMaps, and so they may have null values and this must be checked.
	 * These values are incremented by one unit because at the first step of the
	 * training, we use theses values as counters. Only after this first step,
	 * they are converted to real probabilities and still after that they are
	 * converted to log of probabilities.
	 * 
	 * @param stateFtr
	 * @param emission
	 */
	public void incProbEmissionByFeature(int stateFtr, int emission) {
		int state = featureToState.get(stateFtr);

		HashMap<Integer, Double> emissionMap = probEmission.get(state);
		if (emissionMap == null) {
			emissionMap = new HashMap<Integer, Double>();
			probEmission.set(state, emissionMap);
		}

		Double prob = emissionMap.get(emission);
		if (prob == null)
			prob = 0.0;
		emissionMap.put(emission, prob + 1);
	}

	public void incProbEmissionByFeature(int stateFtr, int emission,
			double value) {

		int state = featureToState.get(stateFtr);

		HashMap<Integer, Double> emissionMap = probEmission.get(state);
		if (emissionMap == null) {
			emissionMap = new HashMap<Integer, Double>();
			probEmission.set(state, emissionMap);
		}

		Double prob = emissionMap.get(emission);
		if (prob == null)
			prob = 0.0;
		emissionMap.put(emission, prob + value);
	}

	public void setProbEmission(int state, int emission, double prob) {
		HashMap<Integer, Double> emissionMap = probEmission.get(state);
		if (emissionMap == null) {
			emissionMap = new HashMap<Integer, Double>();
			probEmission.set(state, emissionMap);
		}
		emissionMap.put(emission, prob);
	}

	public void setProbEmissionByFeature(int stateFtr, int emission, double prob) {

		int state = featureToState.get(stateFtr);

		HashMap<Integer, Double> emissionMap = probEmission.get(state);
		if (emissionMap == null) {
			emissionMap = new HashMap<Integer, Double>();
			probEmission.set(state, emissionMap);
		}
		emissionMap.put(emission, prob);
	}

	/**
	 * Normalize the probability distributions to sum 1.
	 */
	public void normalizeProbabilities() {
		normalizeInitialAndFinalProbabilities();
		normalizeTransitionProbabilities();
		normalizeEmissionProbabilities();
		normalized = true;
	}

	/**
	 * Normalize the transition distributions.
	 */
	public void normalizeTransitionProbabilities() {
		for (int state = 0; state < numStates; ++state)
			normalizeArray(probTransition[state]);
	}

	/**
	 * Normalize the emission distributions.
	 */
	public void normalizeEmissionProbabilities() {
		for (int state = 0; state < numStates; ++state) {
			// Skip unseen states.
			HashMap<Integer, Double> emissionMap = probEmission.get(state);
			if (emissionMap == null)
				continue;

			// Sum of counters within the current state.
			double sum = 0.0;
			for (Double val : emissionMap.values())
				sum += val;

			// Normalize the values by the sum.
			for (Entry<Integer, Double> emission : emissionMap.entrySet()) {
				double prob = emission.getValue() / sum;
				emission.setValue(prob);
			}
		}
	}

	/**
	 * Normalize the initial and final state distributions.
	 */
	public void normalizeInitialAndFinalProbabilities() {
		// Initial probabilities.
		normalizeArray(probInitialState);

		// Final probabilities.
		normalizeArray(probFinalState);
	}

	/**
	 * Calculate the log of the probability parameters to be used in the Viterbi
	 * algorithm.
	 */
	public void applyLog() {
		// Allocate the structures.
		lnProbInitialState = new double[numStates];
		lnProbFinalState = new double[numStates];
		lnProbTransition = new double[numStates][numStates];
		lnProbEmission = new Vector<HashMap<Integer, Double>>(numStates);
		lnProbEmission.setSize(numStates);

		// Initial state probabilities.
		applyLog(probInitialState, lnProbInitialState);

		// Final state probabilities.
		applyLog(probFinalState, lnProbFinalState);

		// Clear emission log-probability maps.
		lnProbEmission.clear();
		lnProbEmission.setSize(numStates);

		for (int state = 0; state < numStates; ++state) {
			// Transition probabilities.
			applyLog(probTransition[state], lnProbTransition[state]);

			// Skip unseen states.
			HashMap<Integer, Double> emissionMap = probEmission.get(state);
			if (emissionMap == null)
				continue;

			// Emission probabilities.
			HashMap<Integer, Double> lnEmissionMap = new HashMap<Integer, Double>();
			lnProbEmission.set(state, lnEmissionMap);

			// Apply the log for all values within the map.
			for (Entry<Integer, Double> emission : emissionMap.entrySet())
				lnEmissionMap.put(emission.getKey(), log(emission.getValue()));
		}
	}

	/**
	 * Modified log function. Return Double.NEGATIVE_INFINITY if the input is
	 * zero or less.
	 * 
	 * @param val
	 * @return
	 */
	private double log(double val) {
		if (val <= 0.0)
			return Double.NEGATIVE_INFINITY;
		return Math.log(val);
	}

	/**
	 * Normalize the elements of the given array so that they represent a
	 * probability distribution.
	 * 
	 * @param values
	 *            an array of values that represents some unnormalized
	 *            probability distribution.
	 * @param smooth
	 * @param lnValues
	 *            the corresponding log-array where the log of the calculated
	 *            probabilities are stored.
	 */
	protected void normalizeArray(double[] values) {
		// Calculate the sum of all values within the array.
		double sum = 0.0;
		for (double val : values)
			sum += val;

		if (sum <= 0.0)
			return;

		// Normalize by the sum.
		for (int idx = 0; idx < values.length; ++idx)
			values[idx] /= sum;
	}

	private double mean(Iterable<Double> values) {
		int count = 0;
		double sum = 0.0;
		for (Double val : values) {
			sum += val;
			++count;
		}
		return sum / count;
	}

	private double mean(double[] values) {
		int count = 0;
		double sum = 0.0;
		for (Double val : values) {
			sum += val;
			++count;
		}
		return sum / count;
	}

	private void applyLog(double[] values, double[] lnValues) {
		for (int idx = 0; idx < values.length; ++idx)
			lnValues[idx] = log(values[idx]);
	}

	/**
	 * Clear the model parameters to start a new training procedure.
	 */
	public void clearParameters() {
		probEmission.clear();
		probEmission.setSize(numStates);

		for (int st = 0; st < numStates; ++st) {
			probInitialState[st] = 0.0;
			probFinalState[st] = 0.0;

			for (int stTo = 0; stTo < numStates; ++stTo)
				probTransition[st][stTo] = 0.0;
		}
	}

	/**
	 * Save the trained model to a new file with the given name, using an
	 * independent representation.
	 * 
	 * @param fileName
	 *            the name of the file to store the model.
	 * 
	 * @throws FileNotFoundException
	 *             if some problem happens when opening the file.
	 */
	public void save(String fileName) throws FileNotFoundException {
		PrintStream ps = new PrintStream(fileName);
		save(ps);
		ps.close();
	}

	public void load(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		load(reader);
		reader.close();
	}

	public void load(InputStream is) throws IOException {
		load(new BufferedReader(new InputStreamReader(is)));
	}

	public void load(BufferedReader reader) throws IOException {
		// List of states.
		LinkedList<Integer> states = new LinkedList<Integer>();

		// State set.
		String buff = skipComments(reader);
		if (stateToFeature == null) {
			if (buff == null)
				throw new IOException("Empty model");

			for (String label : buff.split("\\s"))
				states.add(featureValueEncoding.putString(label));

			// Set the feature values that are states.
			initStateFeatures(states);
		}

		// Create the model parameters.
		allocProbabilityParameters();

		// Initial and final states probabilities.
		for (int pos = 0; pos < numStates; ++pos) {
			String[] labelAndProbs = skipComments(reader).split("\\s");
			int state = getStateByString(labelAndProbs[0]);
			double probInitial = Double.parseDouble(labelAndProbs[1]);
			double probFinal = Double.parseDouble(labelAndProbs[2]);
			probInitialState[state] = probInitial;
			probFinalState[state] = probFinal;
		}

		// Transition probabilities.
		for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
			String[] labelsAndProbs = skipComments(reader).split("\\s");
			for (int pos = 0; pos < numStates; ++pos) {
				int stateTo = getStateByString(labelsAndProbs[pos * 2]);
				double prob = Double.parseDouble(labelsAndProbs[pos * 2 + 1]);
				probTransition[stateFrom][stateTo] = prob;
			}
		}

		// Emission probabilities.
		for (int state = 0; state < numStates; ++state) {
			String[] emissionLabelAndProb = skipComments(reader).split("\\s");
			for (int pos = 0; pos < emissionLabelAndProb.length; pos += 2) {
				String label = emissionLabelAndProb[pos];
				double prob = Double.parseDouble(emissionLabelAndProb[pos + 1]);
				int emission = featureValueEncoding.putString(label);
				setProbEmission(state, emission, prob);
			}
		}
	}

	/**
	 * Load a model generated by a Hadoop trainer.
	 * 
	 * @param fileName
	 *            name of a file that contains a model generated by a Hadoop
	 *            trainer algorithm.
	 * @throws IOException
	 */
	public void loadFromHadoopModel(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		loadFromHadoopModel(reader);
		reader.close();
	}

	/**
	 * Load a model generated by a Hadoop trainer.
	 * 
	 * @param reader
	 * @throws IOException
	 */
	public void loadFromHadoopModel(BufferedReader reader) throws IOException {
		HmmModelStrings sparseModel = new HmmModelStrings(reader);

		if (stateToFeature == null) {
			// Fill the list of states (guarantee that the OUT state is the
			// first).
			LinkedList<Integer> states = new LinkedList<Integer>();
			states.add(featureValueEncoding
					.putString(HmmModelStrings.OUT_STATE));
			for (Entry<String, HashMap<String, Double>> entry : sparseModel
					.getTransitionIterable()) {
				String state = entry.getKey();
				if (state.equals(HmmModelStrings.OUT_STATE))
					continue;
				states.add(featureValueEncoding.putString(state));
			}

			// Set the feature values that are states.
			initStateFeatures(states);
		}

		// Create the model parameters.
		allocProbabilityParameters();

		// Initial and final states probabilities.
		for (Entry<String, Double> probs : sparseModel.getFirstStateIterable()) {
			String stateLabel = probs.getKey();
			double probInitial = probs.getValue();
			int state = getStateByString(stateLabel);
			probInitialState[state] = probInitial;
			probFinalState[state] = 1.0 / getNumberOfStates();
		}

		// Transition probabilities.
		for (Entry<String, HashMap<String, Double>> probsMap : sparseModel
				.getTransitionIterable()) {
			int stateFrom = getStateByString(probsMap.getKey());
			for (Entry<String, Double> probs : probsMap.getValue().entrySet()) {
				int stateTo = getStateByString(probs.getKey());
				probTransition[stateFrom][stateTo] = probs.getValue();
			}
		}

		// Emission probabilities.
		for (Entry<String, HashMap<String, Double>> probsMap : sparseModel
				.getEmissionIterable()) {
			int state = getStateByString(probsMap.getKey());
			for (Entry<String, Double> probs : probsMap.getValue().entrySet()) {
				String symbolLabel = probs.getKey();
				double prob = probs.getValue();
				int symbol = featureValueEncoding.putString(symbolLabel);
				setProbEmission(state, symbol, prob);
			}
		}

		// Calculate the log of the probabilities.
		applyLog();
	}

	/**
	 * Skip comment lines and blank lines in the given
	 * <code>BufferedReader</code> and returns the first found line with valid
	 * content.
	 * 
	 * @param reader
	 * @return
	 * @throws IOException
	 */
	private String skipComments(BufferedReader reader) throws IOException {
		String buff = null;
		while ((buff = reader.readLine()) != null) {
			// Skip comments.
			if (buff.startsWith("#"))
				continue;
			// Skip blank lines.
			if (buff.trim().length() == 0)
				continue;
			// A valid line.
			break;
		}

		return buff;
	}

	/**
	 * Save the trained model using an independent representation to the given
	 * stream.
	 * 
	 * Save only the probability parameters, i.e., do not save their log values
	 * that can be easily calculated again after a model is loaded.
	 * 
	 * @param out
	 *            the output stream where to write the values.
	 */
	public void save(PrintStream out) {
		out.println("# HMM model");
		out.println();

		out.println("# States");
		for (int ftrVal : stateToFeature)
			out.print(getFeatureString(ftrVal) + " ");
		out.println();

		out.println("# Initial and terminal state probabilities");
		for (int st = 0; st < numStates; ++st)
			out.println(getStateLabel(st) + " " + probInitialState[st] + " "
					+ probFinalState[st]);
		out.println();

		out.println("# Transition probabilities");
		for (int stFrom = 0; stFrom < numStates; ++stFrom) {
			out.println("# State: " + getStateLabel(stFrom));
			for (int stTo = 0; stTo < numStates; ++stTo)
				out.print(getStateLabel(stTo) + " "
						+ probTransition[stFrom][stTo] + " ");
			out.println();
		}
		out.println();

		out.println("# Emission probabilities");
		for (int st = 0; st < numStates; ++st) {
			HashMap<Integer, Double> emissionMap = probEmission.get(st);
			if (emissionMap == null)
				continue;
			out.println("# State: " + getStateLabel(st));
			for (Entry<Integer, Double> emission : emissionMap.entrySet())
				out.print(getFeatureString(emission.getKey()) + " "
						+ emission.getValue() + " ");
			out.println();
		}
		out.println();
	}

	/**
	 * Return the label of the state with the given index.
	 * 
	 * @param state
	 *            a state index.
	 * 
	 * @return the state label.
	 */
	public String getStateLabel(int state) {
		return featureValueEncoding.getLabelByCode(stateToFeature[state]);
	}

	/**
	 * Return the string representation of the given feature value.
	 * 
	 * @param featureValue
	 *            a feature value within the used
	 *            <code>FeatureValueEncoding</code>.
	 * 
	 * @return the string feature representation.
	 */
	public String getFeatureString(int featureValue) {
		return featureValueEncoding.getLabelByCode(featureValue);
	}

	public int getStateByString(String stateLabel) {
		int stateFtr = featureValueEncoding.getCodeByLabel(stateLabel);
		if (stateFtr < 0 || stateFtr == FeatureValueEncoding.UNSEEN_CODE)
			return -1;
		return featureToState.get(stateFtr);
	}

	public int getNumberOfStates() {
		return numStates;
	}

	public Map<Integer, Double> getEmissionMap(int state) {
		return probEmission.get(state);
	}

	public FeatureValueEncoding getFeatureValueEncoding() {
		return featureValueEncoding;
	}

	/**
	 * Return the number of transition probability parameters greater than zero.
	 * 
	 * @return
	 */
	public int getNumberOfTransitions() {
		int count = 0;
		for (int stateFrom = 0; stateFrom < numStates; ++stateFrom)
			for (int stateTo = 0; stateTo < numStates; ++stateTo)
				if (probTransition[stateFrom][stateTo] > 0d)
					++count;
		return count;
	}

	/**
	 * Return the number of emission probability parameters greater than zero.
	 * 
	 * @return
	 */
	public int getNumberOfEmissions() {
		int count = 0;
		for (Map<Integer, Double> emissionMap : probEmission)
			count += emissionMap.size();
		return count;
	}

	/**
	 * Return the minimum emission probability that is greater than zero.
	 * 
	 * @return
	 */
	public double getMinimumEmissionProbability() {
		double min = Double.MAX_VALUE;
		for (Map<Integer, Double> emissionMap : probEmission)
			for (Double prob : emissionMap.values())
				if (prob > 0d && prob < min)
					min = prob;
		return min;
	}

	/**
	 * Remove emission probability values that have value equal to zero.
	 */
	public void removeZeroEmissionProbabilities() {
		LinkedList<Integer> keysToRemove = new LinkedList<Integer>();
		for (Map<Integer, Double> emissionMap : probEmission) {
			for (Entry<Integer, Double> entry : emissionMap.entrySet())
				if (entry.getValue() <= 0d)
					keysToRemove.add(entry.getKey());
			Iterator<Integer> it = keysToRemove.iterator();
			while (it.hasNext()) {
				emissionMap.remove(it.next());
				it.remove();
			}
		}
	}

	/**
	 * Create explicit zero probability emissions.
	 */
	public void setImplicitZeroEmissionProbabilities() {
		for (int state = 0; state < numStates; ++state) {
			for (Integer symbol : featureValueEncoding.getCollectionOfLabels())
				if (probEmission.get(state).get(symbol) == null)
					probEmission.get(state).put(symbol, 0d);
		}
	}

	/**
	 * Apply (or not) a smoothing technique to this model. This method expects
	 * the model has not been normalized yet, i.e., it still contains the
	 * occurrence counters, since most smoothing techniques use the counters.
	 * 
	 * @param type
	 * @throws HmmException
	 */
	public void applySmoothing(Smoothing type) throws HmmException {
		if (normalized)
			throw new HmmException(
					"One can not apply a smoothing technique to a normalized model.");

		switch (type) {
		case NONE:
			applyNoSmoothing();
			break;
		case LAPLACE:
			applyLaplaceSmoothing();
			break;
		case ABSOLUTE_DISCOUNTING:
			applyAbsoluteDiscountSmoothing();
			break;
		}

		// Normalize the initial, final and transition distributions and apply
		// log to use in tagging.
		normalizeInitialAndFinalProbabilities();
		normalizeTransitionProbabilities();
		applyLog();
	}

	/**
	 * Just normalize the counters, i.e., do not apply any smoothing.
	 */
	private void applyNoSmoothing() {
		normalizeEmissionProbabilities();
	}

	/**
	 * Apply to this model the absolute discount smoothing technique.
	 */
	private void applyAbsoluteDiscountSmoothing() {
		Collection<Integer> symbols = featureValueEncoding
				.getCollectionOfLabels();
		HashMap<Integer, Double> probDistribution;
		for (int state = 0; state < numStates; ++state) {
			probDistribution = probEmission.get(state);
			if (probDistribution == null) {
				probDistribution = new HashMap<Integer, Double>();
				probEmission.set(state, probDistribution);
			}

			// Count the number of occurences of this state.
			double sum = 0;
			int numberOfSeenSymbolsInThisState = 0;
			for (int symbol : symbols) {
				double count = getEmissionProbability(symbol, state);
				sum += count;
				if (count > 0d)
					++numberOfSeenSymbolsInThisState;
			}

			// Apply the smoothing and normalized at the same time.
			int numberOfSymbols = featureValueEncoding.size();
			double discountValue = 1e-6 / numberOfSeenSymbolsInThisState;
			double probUnseen = discountValue * numberOfSeenSymbolsInThisState
					/ (numberOfSymbols - probDistribution.size());

			for (int symbol : symbols) {
				Double count = probDistribution.get(symbol);
				if (count == null)
					probDistribution.put(symbol, probUnseen);
				else if (count > 0d)
					probDistribution.put(symbol, (count / sum) - discountValue);
			}
		}
	}

	/**
	 * Apply one of the most used and simple smoothing techniques: Laplace
	 * smoothing. Just sum one to each counter and then normalize. This method
	 * assumes that the values in the emission probability parameters of this
	 * model are in fact the occurence counters, i.e., they are not normalized
	 * yet.
	 */
	private void applyLaplaceSmoothing() {
		Collection<Integer> symbols = featureValueEncoding
				.getCollectionOfLabels();
		HashMap<Integer, Double> probDistribution;
		for (int state = 0; state < numStates; ++state) {
			probDistribution = probEmission.get(state);
			if (probDistribution == null) {
				probDistribution = new HashMap<Integer, Double>();
				probEmission.set(state, probDistribution);
			}

			// Sum 1 to all the probability values.
			double ALPHA = 0.01;
			for (int symbol : symbols) {
				Double prob = probDistribution.get(symbol);
				if (prob == null)
					probDistribution.put(symbol, ALPHA);
				else if (prob > 0d)
					probDistribution.put(symbol, prob + ALPHA);
			}
		}

		normalizeEmissionProbabilities();
	}
}
