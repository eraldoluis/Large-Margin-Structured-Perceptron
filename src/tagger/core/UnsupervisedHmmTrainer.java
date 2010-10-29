package tagger.core;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Vector;

import tagger.data.Dataset;
import tagger.data.DatasetExample;
import tagger.data.DatasetException;
import tagger.data.FeatureValueEncoding;
import tagger.utils.RandomGenerator;

/**
 * Generative-unsupervised HMM trainer.
 * 
 * The training algorithm is based on Baum-Welch algorithm (forward-backward).
 * 
 * @author eraldof
 * 
 */
public class UnsupervisedHmmTrainer {
	/**
	 * Index of the feature used as observation value.
	 */
	protected int observationFeature;

	/**
	 * The trainset.
	 */
	protected Dataset trainset;

	/**
	 * Current model.
	 */
	protected HmmModel model;

	/**
	 * Number of states of the current model.
	 */
	protected int numStates;

	/**
	 * Numerator values of the new initial state probability distribution.
	 */
	protected Vector<Double> probInitialStateNew;

	/**
	 * Normalization factor (denominator) of the new initial state probability
	 * distribution.
	 */
	protected double normFactorInitialState;

	/**
	 * Numerator values of the new transition probability distributions.
	 */
	protected Vector<Vector<Double>> probTransitionNew;

	/**
	 * Normalization factor (denominator) of the new transition probability
	 * distributions.
	 */
	protected double[] normFactorTransition;

	/**
	 * Numerator of the new emission probability distributions.
	 */
	protected Vector<HashMap<Integer, Double>> probEmissionNew;

	/**
	 * Normalization factor (denominator) for the emission probability
	 * distributions.
	 */
	protected double[] normFactorEmission;

	/**
	 * Temporary variables: alpha[t][i] = p(o_1, ..., o_t, y_t=i | model).
	 */
	protected Vector<Vector<Double>> alpha;

	/**
	 * Temporary variables: beta[t][i] = p(o_t+1, ..., o_T | y_t=i, model).
	 */
	protected Vector<Vector<Double>> beta;

	/**
	 * Weight for each example. Each weight may be either just a
	 * <code>Double</code>, i.e., the same weight for every token within the
	 * example or a <code>VectorDouble</code> with a weight for each token.
	 */
	protected Vector<Object> weights;

	/**
	 * Weight array for the current example to faster access.
	 */
	protected double[] curExampleWeights;

	/**
	 * Default constructor.
	 */
	public UnsupervisedHmmTrainer() {
	}

	/**
	 * Set the initial model.
	 * 
	 * This model is used in the first iteration of the Baum-Welch algorithm. It
	 * gives a strong bias to the final solution.
	 * 
	 * @param iniModel
	 */
	public void setInitialModel(HmmModel iniModel) {
		model = iniModel;
		if (model != null)
			numStates = model.getNumberOfStates();
	}

	/**
	 * Set the weight vector that indicates the weight of each example in the
	 * trainset given to the next call of the train
	 * 
	 * @param weights
	 */
	public void setWeights(Collection<Object> weights) {
		this.weights = new Vector<Object>(weights);
	}

	/**
	 * Train an HMM model using the Baum-Welch (forward-backward) algorithm and
	 * the observation sequences in the given training set.
	 * 
	 * @param trainset
	 *            a set of examples with an observation feature.
	 * @param observationFeatureLabel
	 *            the label of the observation feature in the dataset.
	 * @param stateFeatureLabels
	 *            an array of strings given all the state labels to be used.
	 * @param numIterations
	 *            the number of iterations of the BW algorithm.
	 * 
	 * @return the trained model.
	 * 
	 * @throws DatasetException
	 *             if the dataset doest not contain the given observation
	 *             feature label.
	 * @throws HmmException
	 */
	public HmmModel train(Dataset trainset, String observationFeatureLabel,
			String[] stateFeatureLabels, int numIterations)
			throws DatasetException, HmmException {

		this.trainset = trainset;

		this.observationFeature = trainset
				.getFeatureIndex(observationFeatureLabel);
		if (observationFeature < 0)
			throw new DatasetException("Observation feature "
					+ observationFeatureLabel + " does not exist.");

		// Feature-value encoding.
		FeatureValueEncoding fve = trainset.getFeatureValueEncoding();

		if (model == null) {
			// Find the set of states within the state feature.
			Vector<Integer> stateFeatures = new Vector<Integer>();
			for (String stateFtrLabel : stateFeatureLabels)
				stateFeatures.add(fve.putString(stateFtrLabel));

			// Create an empty model.
			model = new HmmModel(fve, stateFeatures);
			numStates = model.getNumberOfStates();
			// Init the model with random distributions.
			initModel();
		} else if (model.getFeatureValueEncoding() != trainset
				.getFeatureValueEncoding())
			throw new DatasetException(
					"The initial model has a different feature-value encoding than the given trainset.");

		// Allocate and initialize the normalization factor variables.
		normFactorEmission = new double[numStates];
		normFactorTransition = new double[numStates];

		// Allocate and initialize the probability accumulators.
		probTransitionNew = new Vector<Vector<Double>>(numStates);
		probTransitionNew.setSize(numStates);
		probInitialStateNew = new Vector<Double>(numStates);
		probInitialStateNew.setSize(numStates);
		probEmissionNew = new Vector<HashMap<Integer, Double>>(numStates);
		probEmissionNew.setSize(numStates);
		for (int state = 0; state < numStates; ++state) {
			Vector<Double> vd = new Vector<Double>(numStates);
			vd.setSize(numStates);
			probTransitionNew.set(state, vd);
		}

		double prevLogProbObservations = Double.NEGATIVE_INFINITY;

		// Baum-Welch iterations.
		for (int iter = 0; iter < numIterations; ++iter) {
			// Stub.
			preIteration(iter);

			// Initialize the accumulators for the probability estimation.
			initIteration();

			int impossibleExamples = 0;
			double logProbObservations = 0.0;

			// Train over each observation sequence in the training set.
			for (DatasetExample example : trainset) {
				// Adjust the length of the temporary variables.
				adjustTempVariablesSizes(example);

				// Forward and backward algorithm.
				forward(example);
				backward(example);

				// Estimate the counters of transitions and emission for this
				// example.
				double probObservation = accountExampleOnProbabilityEstimates(example);
				if (probObservation <= 0.0)
					++impossibleExamples;
				else
					logProbObservations += Math.log(probObservation);
			}

			// TODO just for debuging
			System.out
					.println("Log-likelihood on iteration "
							+ iter
							+ " is "
							+ logProbObservations
							+ " ("
							+ impossibleExamples
							+ (prevLogProbObservations != Double.NEGATIVE_INFINITY ? ", "
									+ (logProbObservations - prevLogProbObservations)
									+ ")"
									: ")"));

			if (logProbObservations < prevLogProbObservations)
				System.err.println("WARNING! Log-likelihood incresead!");

			prevLogProbObservations = logProbObservations;

			// Estimate the new model probabilities from the accumulators.
			normalizeProbabiltiesAndUpdateModel();

			// Stub to some process in the derived classes.
			postIteration(iter);
		}

		// Apply log to the probability values.
		model.applyLog();

		return model;
	}

	protected void preIteration(int iter) throws DatasetException {
	}

	protected void postIteration(int iter) throws DatasetException {
	}

	// Fill the model with uniform probabilities.
	protected void initModel() {
		Random rand = RandomGenerator.gen;

		for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
			// Initial state probabilities.
			model.probInitialState[stateFrom] = rand.nextDouble() + 0.1;

			// Transition probabilities.
			for (int stateTo = 0; stateTo < numStates; ++stateTo)
				model.probTransition[stateFrom][stateTo] = rand.nextDouble() + 0.1;

			// Emission probabilities.
			Collection<Integer> symbols = model.getFeatureValueEncoding()
					.getCollectionOfSymbols();
			HashMap<Integer, Double> emissionMap = new HashMap<Integer, Double>();
			model.probEmission.set(stateFrom, emissionMap);
			for (Integer symbol : symbols)
				emissionMap.put(symbol, rand.nextDouble() + 0.1);
		}

		model.normalizeProbabilities();
	}

	/**
	 * Initialize the variables used to accumulated the values relative to a
	 * complete iteration. These values are counter over the training examples.
	 */
	protected void initIteration() {
		// Initial state normalization factor.
		normFactorInitialState = 0.0;

		for (int state = 0; state < numStates; ++state) {
			// Initial state accumulators.
			probInitialStateNew.set(state, 0.0);

			// Transition accumulators.
			normFactorTransition[state] = 0.0;
			for (int stateTo = 0; stateTo < numStates; ++stateTo)
				probTransitionNew.get(state).set(stateTo, 0.0);

			// On every iteration, create a new emission map, instead of
			// iniatilizing it. This new emission map will be used by the model
			// at the end of the iteration.
			probEmissionNew.set(state, new HashMap<Integer, Double>());
			normFactorEmission[state] = 0.0;
		}
	}

	/**
	 * Adjust the sizes of the alpha, beta, gamma and xi variables to
	 * accommodate the calculation for the given example. Also allocated and
	 * fill the weight array for the current example sequence.
	 * 
	 * @param example
	 *            an example that will be considered in the next iteration.
	 */
	protected void adjustTempVariablesSizes(DatasetExample example) {
		int exSize = example.size();

		// Organize the weights for the given example so that we have a double
		// array with a weight for each token of the currect example.
		if (curExampleWeights == null || curExampleWeights.length < exSize)
			curExampleWeights = new double[exSize];

		if (weights == null)
			for (int tkn = 0; tkn < exSize; ++tkn)
				curExampleWeights[tkn] = 1.0;
		else {
			Object obj = weights.get(example.getIndex());
			if (obj instanceof Double) {
				double fixedWeight = (Double) obj;
				for (int tkn = 0; tkn < exSize; ++tkn)
					curExampleWeights[tkn] = fixedWeight;
			} else if (obj instanceof Vector<?>) {
				@SuppressWarnings("unchecked")
				Vector<? extends Double> curExampleWeightsV = (Vector<? extends Double>) weights
						.get(example.getIndex());
				for (int tkn = 0; tkn < exSize; ++tkn)
					curExampleWeights[tkn] = curExampleWeightsV.get(tkn);
			}
		}

		if (alpha == null) {
			alpha = new Vector<Vector<Double>>(exSize);
			beta = new Vector<Vector<Double>>(exSize);
		}

		int prevSize = alpha.size();
		if (prevSize >= exSize)
			return;

		// Expand vectors.
		alpha.setSize(exSize);
		beta.setSize(exSize);

		// Create new elements for the extra tokens in the current example.
		for (int token = prevSize; token < exSize; ++token) {
			Vector<Double> vd;

			vd = new Vector<Double>(numStates);
			vd.setSize(numStates);
			alpha.set(token, vd);

			vd = new Vector<Double>(numStates);
			vd.setSize(numStates);
			beta.set(token, vd);
		}
	}

	/**
	 * Forward algorithm that fills the alpha variables.
	 * 
	 * @param example
	 *            sequence for what the alpha variables are calculated.
	 */
	protected void forward(DatasetExample example) {
		int size = example.size();

		// Initialize the alpha[0] values.
		Vector<Double> first = alpha.get(0);
		for (int state = 0; state < numStates; ++state) {
			int emission = example.getFeatureValue(0, observationFeature);
			double emisProb = model.getEmissionProbability(emission, state)
					* model.getInitialStateProbability(state);
			first.set(state, emisProb);
		}

		for (int token = 1; token < size; ++token) {
			for (int stateTo = 0; stateTo < numStates; ++stateTo) {
				// Probability of arriving at this state (without emission).
				double prevTokenProb = 0.0;
				for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
					double transProb = model.getTransitionProbability(
							stateFrom, stateTo);
					prevTokenProb += transProb
							* alpha.get(token - 1).get(stateFrom);
				}

				// Complete probability of arriving in this state and emits the
				// current symbol.
				int emission = example.getFeatureValue(token,
						observationFeature);
				double emisProb = model.getEmissionProbability(emission,
						stateTo);
				alpha.get(token).set(stateTo, emisProb * prevTokenProb);
			}
		}
	}

	/**
	 * Backward algorithm that fills the beta variables.
	 * 
	 * @param example
	 *            sequence used to fill the beta variables.
	 */
	protected void backward(DatasetExample example) {
		int size = example.size();

		// Initialize the beta[size-1] values.
		Vector<Double> last = beta.get(size - 1);
		for (int state = 0; state < numStates; ++state)
			last.set(state, 1.0);

		for (int token = size - 2; token >= 0; --token) {
			for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
				double betaVal = 0.0;
				int emission = example.getFeatureValue(token + 1,
						observationFeature);

				for (int stateTo = 0; stateTo < numStates; ++stateTo) {
					// Probability of emiting the next symbol.
					double emisProb = model.getEmissionProbability(emission,
							stateTo);

					// Probability of transition.
					double transProb = model.getTransitionProbability(
							stateFrom, stateTo);

					betaVal += beta.get(token + 1).get(stateTo) * emisProb
							* transProb;
				}

				beta.get(token).set(stateFrom, betaVal);
			}
		}
	}

	int myid = -1;
	int mysize = 100000000;

	/**
	 * Account the given example in the estimation of the new model
	 * probabilities.
	 * 
	 * @param example
	 * 
	 * @return the probability of the given example be generated by the current
	 *         model.
	 * @throws HmmException
	 */
	protected double accountExampleOnProbabilityEstimates(DatasetExample example)
			throws HmmException {
		int size = example.size();

		// Calculate the probability of the current observation sequence given
		// the model: p(Y | model). This is the denominator necessary to
		// normalize all the gamma's and xi's values.
		double probObservation = 0.0;
		for (int state = 0; state < numStates; ++state)
			probObservation += alpha.get(size - 1).get(state);

		// If the example has zero probability, just skip it.
		if (probObservation <= 0.0) {
			if (size < mysize) {
				mysize = size;
				myid = example.getIndex();
			}
			return probObservation;
		}

		for (int token = 0; token < size; ++token) {
			// Emission symbol in the current token.
			int curSymbol = example.getFeatureValue(token, observationFeature);

			// Emission symbol in the next token.
			int nextSymbol = -1;
			if (token < size - 1)
				nextSymbol = example.getFeatureValue(token + 1,
						observationFeature);

			for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {

				// Gamma numerator value.
				double gammaVal = alpha.get(token).get(stateFrom)
						* beta.get(token).get(stateFrom);

				// Normalize by the probability of the current observation
				// sequence.
				gammaVal /= probObservation;

				// Scale the gamma value with the weight of the current example
				// before accounting it in the parameter reestimation.
				if (curExampleWeights[token] != 1.0)
					gammaVal *= curExampleWeights[token];

				// Update the emission probability (numerator).
				incProbabilityEmissionNew(stateFrom, curSymbol, gammaVal);

				// Update the normalization factor for the emission
				// probabilities from the current state.
				normFactorEmission[stateFrom] += gammaVal;

				// Update the initial state probability for the current state.
				if (token == 0) {
					// Normalization factor.
					normFactorInitialState += gammaVal;
					// Probability numerator.
					double prevProb = probInitialStateNew.get(stateFrom);
					probInitialStateNew.set(stateFrom, gammaVal + prevProb);
				}

				// Xi values: transition probability distributions. Only
				// calculate them if this is not the last token.
				if (token < size - 1) {
					for (int stateTo = 0; stateTo < numStates; ++stateTo) {
						// Xi numerator value.
						double xiVal = alpha.get(token).get(stateFrom)
								* model.getTransitionProbability(stateFrom,
										stateTo)
								* model.getEmissionProbability(nextSymbol,
										stateTo)
								* beta.get(token + 1).get(stateTo);

						// Normalize by the probability of the current
						// observation sequence.
						xiVal /= probObservation;

						// Scale the Xi value with the weight of the current
						// example before accounting it in the parameter
						// reestimation.
						if (curExampleWeights[token] != 1.0)
							xiVal *= curExampleWeights[token];

						// Update the normalization factor for the transition
						// probabilities from the current state.
						normFactorTransition[stateFrom] += xiVal;

						// Update the transition probability accumulator
						// (numerator) for this (token, stateFrom and stateTo)
						// tuple.
						double prevProb = probTransitionNew.get(stateFrom).get(
								stateTo);
						probTransitionNew.get(stateFrom).set(stateTo,
								prevProb + xiVal);
					}
				}
			}
		}

		return probObservation;
	}

	/**
	 * Normalize the accumulated counters and update the current model.
	 */
	protected void normalizeProbabiltiesAndUpdateModel() {
		for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
			// Initial state probabilities.
			if (normFactorInitialState > 0.0)
				model.probInitialState[stateFrom] = probInitialStateNew
						.get(stateFrom) / normFactorInitialState;
			else
				model.probInitialState[stateFrom] = 0.0;

			// Transition probabilities.
			if (normFactorTransition[stateFrom] > 0.0)
				for (int stateTo = 0; stateTo < numStates; ++stateTo)
					model.probTransition[stateFrom][stateTo] = probTransitionNew
							.get(stateFrom).get(stateTo)
							/ normFactorTransition[stateFrom];
			else
				for (int stateTo = 0; stateTo < numStates; ++stateTo)
					model.probTransition[stateFrom][stateTo] = 0.0;

			// Emission probabilities.
			HashMap<Integer, Double> emissionMapNew = probEmissionNew
					.get(stateFrom);
			if (normFactorEmission[stateFrom] > 0.0)
				for (Entry<Integer, Double> emission : emissionMapNew
						.entrySet())
					emission.setValue(emission.getValue()
							/ normFactorEmission[stateFrom]);
			else
				for (Entry<Integer, Double> emission : emissionMapNew
						.entrySet())
					emission.setValue(0.0);

			// Change the model emission map by the new emission map.
			model.probEmission.set(stateFrom, emissionMapNew);
		}
	}

	/**
	 * Increment the value of the probability of emission.
	 * 
	 * Deal with the case in which the symbol yet does not exist in the emission
	 * map of the state.
	 * 
	 * @param state
	 * @param symbol
	 * @param value
	 */
	protected void incProbabilityEmissionNew(int state, int symbol, double value) {
		HashMap<Integer, Double> emissionMap = probEmissionNew.get(state);
		Double prevValue = emissionMap.get(symbol);
		emissionMap.put(symbol, prevValue == null ? value : prevValue + value);
	}
}
