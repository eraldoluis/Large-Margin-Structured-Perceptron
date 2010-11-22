package tagger.core;

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
import java.util.Random;
import java.util.Map.Entry;
import java.util.Vector;

import org.apache.log4j.Logger;

import tagger.data.Dataset;
import tagger.data.DatasetExample;
import tagger.data.DatasetException;
import tagger.data.FeatureValueEncoding;
import tagger.data.HmmModelStrings;
import tagger.utils.RandomGenerator;

public class HmmModel {
	static Logger logger = Logger.getLogger(HmmModel.class);

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
	 * Probability that is distributed over the unseen symbols in each state.
	 */
	protected double emissionSmoothingProbability;

	/**
	 * Minimal value of every transition probability.
	 */
	protected double transitionSmoothingProbability;

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
		useFinalProbabilities = false;
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
		useFinalProbabilities = false;
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
		useFinalProbabilities = false;
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
		useFinalProbabilities = false;
		this.featureValueEncoding = new FeatureValueEncoding();
		if (fromHadoop)
			loadFromHadoopModel(fileName);
		else
			load(fileName);
	}

	public HmmModel(String fileName, boolean fromHadoop, HmmModel baseModel)
			throws IOException {
		useFinalProbabilities = false;
		this.featureValueEncoding = baseModel.featureValueEncoding;
		this.numStates = baseModel.numStates;
		this.featureToState = baseModel.featureToState;
		this.stateToFeature = baseModel.stateToFeature;
		if (fromHadoop)
			loadFromHadoopModel(fileName);
		else
			load(fileName);
	}

	/**
	 * Initialize the model with the given values (intialState, emissions and
	 * transitions). Use the given feature-value encoding.
	 * 
	 * @param encoding
	 * @param initialState
	 * @param emissions
	 * @param transitions
	 */
	public HmmModel(FeatureValueEncoding encoding,
			HashMap<Integer, Double> initialState,
			HashMap<Integer, HashMap<Integer, Double>> emissions,
			HashMap<Integer, HashMap<Integer, Double>> transitions) {

		featureValueEncoding = encoding;

		// Set of states.
		HashSet<Integer> featuresThatAreState = new HashSet<Integer>(
				initialState.keySet());
		featuresThatAreState.addAll(emissions.keySet());
		featuresThatAreState.addAll(transitions.keySet());

		// Initilize the basic variables.
		initStateFeatures(featuresThatAreState);
		allocProbabilityParameters();

		// Initial state probabilities.
		for (Entry<Integer, Double> entry : initialState.entrySet()) {
			int state = featureToState.get(entry.getKey());
			probInitialState[state] = entry.getValue();
		}

		// Emission probabilities.
		for (Entry<Integer, HashMap<Integer, Double>> entry : emissions
				.entrySet()) {
			int state = featureToState.get(entry.getKey());
			probEmission.set(state, entry.getValue());
		}

		// Transition probabilities.
		for (Entry<Integer, HashMap<Integer, Double>> entry : transitions
				.entrySet()) {
			int stateFrom = featureToState.get(entry.getKey());
			for (Entry<Integer, Double> distEntry : entry.getValue().entrySet()) {
				int stateTo = featureToState.get(distEntry.getKey());
				probTransition[stateFrom][stateTo] = distEntry.getValue();
			}
		}

		// Apply the log to the probabilities (used by the Viterbi algorithm).
		applyLog();
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
		lnProbInitialState = new double[numStates];

		probFinalState = new double[numStates];
		lnProbFinalState = new double[numStates];

		probTransition = new double[numStates][numStates];
		lnProbTransition = new double[numStates][numStates];

		probEmission = new Vector<HashMap<Integer, Double>>(numStates);
		lnProbEmission = new Vector<HashMap<Integer, Double>>(numStates);
		probEmission.setSize(numStates);
		lnProbEmission.setSize(numStates);
	}

	public boolean getUseFinalProbabilities() {
		return useFinalProbabilities;
	}

	public void setUseFinalProbabilities(boolean useFinalProbabilities) {
		this.useFinalProbabilities = useFinalProbabilities;
	}

	/**
	 * Tag the given dataset.
	 * 
	 * @param dataset
	 * @param stateFeature
	 * @param observationFeature
	 * @throws DatasetException
	 */
	public void tag(Dataset dataset, String observationFeatureLabel,
			String stateFeatureLabel) throws DatasetException {
		// Get the indexes of the observation and state features.
		this.observationFeature = dataset
				.getFeatureIndex(observationFeatureLabel);

		this.stateFeature = dataset.getFeatureIndex(stateFeatureLabel);
		if (this.stateFeature < 0)
			this.stateFeature = dataset.createNewFeature(stateFeatureLabel);

		if (dataset.getFeatureValueEncoding() != featureValueEncoding)
			throw new DatasetException(
					"The dataset given for tagging has a different feature-value encoding."
							+ "You need to use the same.");

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
	public void generateExamples(Dataset dataset,
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
	private void generateExample(Dataset dataset, String id,
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

	protected void viterbi(DatasetExample example) {
		int lenExample = example.size();

		double[][] delta = new double[lenExample][numStates];
		int[][] psi = new int[lenExample][numStates];

		// The log probabilities for the first token.
		for (int state = 0; state < numStates; ++state) {
			double emissionWeight = getEmissionParameter(
					example.getFeatureValue(0, observationFeature), state);
			psi[0][state] = -1;
			delta[0][state] = emissionWeight + getInitialStateParameter(state);
		}

		// Apply each step of the Viterb's algorithm.
		for (int tkn = 1; tkn < lenExample; ++tkn)
			for (int state = 0; state < numStates; ++state)
				viterbi(delta, psi, example, tkn, state);

		// Find the best last state.
		int maxState = -1;
		double maxLogProb = Double.NEGATIVE_INFINITY;
		for (int state = 0; state < numStates; ++state) {
			double logProb = delta[lenExample - 1][state];
			if (useFinalProbabilities)
				logProb += getFinalStateParameter(state);

			if (logProb > maxLogProb) {
				maxLogProb = logProb;
				maxState = state;
			}
		}

		if (maxState == -1) {
			maxState = 0;
			maxLogProb = delta[lenExample - 1][0];
			if (useFinalProbabilities)
				maxLogProb += getFinalStateParameter(0);
		}

		// Reconstruct the best path from the best final state, and tag the
		// example.
		decodeExample(example, delta, psi, maxState);
	}

	protected void viterbi(double[][] delta, int[][] psi,
			DatasetExample example, int token, int state) {

		double maxLogProb = Double.NEGATIVE_INFINITY;
		int maxState = -1;
		for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
			double logProb = delta[token - 1][stateFrom]
					+ getTransitionParameter(stateFrom, state);
			if (logProb > maxLogProb) {
				maxLogProb = logProb;
				maxState = stateFrom;
			}
		}

		if (maxState == -1) {
			maxLogProb = delta[token - 1][0] + getTransitionParameter(0, state);
			maxState = 0;
		}

		double emissionLogProb = getEmissionParameter(
				example.getFeatureValue(token, observationFeature), state);

		psi[token][state] = maxState;
		delta[token][state] = maxLogProb + emissionLogProb;
	}

	protected void decodeExample(DatasetExample example, double[][] delta,
			int[][] psi, int finalState) {
		// Example length.
		int len = delta.length;

		int best = finalState;
		example.setFeatureValue(len - 1, stateFeature,
				stateToFeature[finalState]);
		for (int token = len - 1; token > 0; --token) {
			int stateFeatureVal = stateToFeature[psi[token][best]];
			example.setFeatureValue(token - 1, stateFeature, stateFeatureVal);
			best = psi[token][best];
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
		if (lnEmissionMap == null) {
			if (emissionSmoothingProbability > 0.0)
				return Math.log(getEmissionProbability(symbol, state));
			return Double.NEGATIVE_INFINITY;
		}

		Double lnProb = lnEmissionMap.get(symbol);
		if (lnProb == null) {
			if (emissionSmoothingProbability > 0.0)
				return Math.log(getEmissionProbability(symbol, state));
			return Double.NEGATIVE_INFINITY;
		}

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
		if (emissionMap == null) {
			if (emissionSmoothingProbability > 0.0) {
				int numUnseenSymbols = featureValueEncoding.size();
				return emissionSmoothingProbability / numUnseenSymbols;
			}
			return 0.0;
		}

		Double prob = emissionMap.get(symbol);
		if (prob == null) {
			if (emissionSmoothingProbability > 0.0) {
				int numUnseenSymbols = featureValueEncoding.size()
						- emissionMap.size();
				return emissionSmoothingProbability / numUnseenSymbols;
			}
			return 0.0;
		}

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

	/**
	 * Normalize the counters of occurrence to calculate the probabilities and
	 * then take the log.
	 */
	public void normalizeProbabilities() {
		// Initial probabilities.
		normalizeArray(probInitialState, transitionSmoothingProbability);

		// Final probabilities.
		normalizeArray(probFinalState, transitionSmoothingProbability);

		// Transition and emission probabilities.
		for (int state = 0; state < numStates; ++state) {
			normalizeArray(probTransition[state],
					transitionSmoothingProbability);

			// Skip unseen states.
			HashMap<Integer, Double> emissionMap = probEmission.get(state);
			if (emissionMap == null)
				continue;

			// Sum of counters within the current state.
			double sum = 0.0;
			for (Double val : emissionMap.values())
				sum += val;

			if (emissionSmoothingProbability > 0.0)
				sum += emissionSmoothingProbability;

			// Normalize the values by the sum.
			for (Entry<Integer, Double> emission : emissionMap.entrySet()) {
				double prob = emission.getValue() / sum;
				emission.setValue(prob);
			}
		}
	}

	public void applyLog() {
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
	private void normalizeArray(double[] values, double smooth) {
		// Calculate the sum of all values within the array.
		int numZeroValues = 0;
		double sum = 0.0;
		for (double val : values) {
			if (val <= 0.0)
				++numZeroValues;
			sum += val;
		}

		sum += smooth;

		if (sum <= 0.0)
			return;

		// Normalize by the sum.
		for (int idx = 0; idx < values.length; ++idx) {
			if (values[idx] <= 0.0 && smooth > 0.0)
				values[idx] = smooth / numZeroValues;
			values[idx] = values[idx] /= sum;
		}
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

		// Calculate the log of the probabilities.
		applyLog();
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

	public double getEmissionSmoothingProbability() {
		return emissionSmoothingProbability;
	}

	/**
	 * Set the smoothing value for emission probabilities.
	 * 
	 * The given probability is equaly distributed over the unseen symbols in
	 * each state, i.e., if K symbols were not seen in a state i then the
	 * probability of emiting an unseen symbol in this state will be
	 * emissionSmoothingProbability/K. The number of unseen symbols are
	 * extracted from the total symbols in the feature-value encoding mapping at
	 * the moment of normalizing the probabilities. The normalization method
	 * must be called after changing the smoothing value to garantee that the
	 * sum of all emission probabilities on each state will be one.
	 * 
	 * @param prob
	 */
	public void setEmissionSmoothingProbability(double prob) {
		emissionSmoothingProbability = prob;
	}

	/**
	 * Set the smoothing value for transition probabilities.
	 * 
	 * The normalization method MUST be called after setting this value. Only
	 * setting this smooth value has no effect. In the normalization method, the
	 * transition probabilities are updated in the proper way. Moreover, this
	 * method must be called only once after the model has been calculated.
	 * 
	 * @param prob
	 */
	public void setTransitionSmoothingProbability(double prob) {
		transitionSmoothingProbability = prob;
	}
}
