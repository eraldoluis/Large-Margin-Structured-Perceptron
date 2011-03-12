package tagger.core;

import java.util.Map;
import java.util.Vector;

import tagger.data.Dataset;
import tagger.data.DatasetExample;
import tagger.data.DatasetException;
import tagger.evaluation.Evaluation;
import tagger.evaluation.Performance;

/**
 * Generative semi-supervised HMM trainer.
 * 
 * The training algorithm is based on a modified Baum-Welch algorithm
 * (forward-backward). The algorithm takes an input trainset and a flag vector
 * over the tokens of this trainset that indicates which tokens are corrected
 * tagged and which are not.
 * 
 * Basically, the modified forward and backward algorithms prune some paths
 * among all possible state sequences. For instance, if the t-th token is tagged
 * as i, then the algorithms will only consider the state sequences that with
 * the state i in the t-th token.
 * 
 * @author eraldof
 * 
 */
public class SemiSupervisedHmmTrainer extends UnsupervisedHmmTrainer {

	/**
	 * Flag for each example. Each flag may be either just a
	 * <code>Boolean</code>, i.e., the same flag for every token within the
	 * example or a <code>Vector</code> of <code>Boolean</code> with a flag for
	 * each token.
	 */
	protected Vector<Object> taggedExampleFlags;

	/**
	 * Index of the feature that indicates the state.
	 */
	protected int stateFeature;

	/**
	 * Flags of the current example to faster access.
	 */
	protected boolean[] tagged;

	private Dataset testset;

	/**
	 * Default constructor.
	 */
	public SemiSupervisedHmmTrainer() {
	}

	/**
	 * Set the flags for each example (or even each token) in the trainset that
	 * will be given to the train method.
	 * 
	 * The given vector <code>flags</code> have to contain an item to each
	 * example in the trainset. The type of each item must be either
	 * <code>Boolean</code> or <code>Vector</code> of <code>Boolean</code>. If
	 * an example flag is a <code>Boolean</code> value then all tokens within
	 * this example will be trated as tagged tokens. Otherwise, if an example
	 * flag is a Vector of Boolean's then this vector must contain a Boolean for
	 * each token indicating which tokens are tagged and which are not.
	 * 
	 * @param flags
	 */
	public void setTaggedExampleFlags(Vector<Object> flags) {
		this.taggedExampleFlags = flags;
	}

	/**
	 * Train an HMM model using the Baum-Welch (forward-backward) algorithm and
	 * the observation sequences in the given training set.
	 * 
	 * @param trainset
	 *            a set of examples with an observation feature.
	 * @param observationFeatureLabel
	 *            the label of the observation feature in the dataset.
	 * @param stateFeatureLabel
	 *            the of the state feature in the dataset.
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
			String stateFeatureLabel, String[] stateFeatureLabels,
			int numIterations) throws DatasetException, HmmException {

		this.stateFeature = trainset.getFeatureIndex(stateFeatureLabel);
		if (stateFeature < 0)
			throw new DatasetException("State feature " + stateFeatureLabel
					+ " does not exist.");

		return super.train(trainset, observationFeatureLabel,
				stateFeatureLabels, numIterations);
	}

	public void setTuningSet(Dataset dataset) {
		this.testset = dataset;
	}

	protected void preIteration(int iter) throws DatasetException {
		// TODO debug
		System.out.println("# transitions: " + model.getNumberOfTransitions());
		System.out.println("# emission: " + model.getNumberOfEmissions());

		if (iter == 0)
			postIteration(iter);
	}

	@Override
	protected void postIteration(int iter) throws DatasetException {
		// TODO debug
		System.out.println("# transitions: " + model.getNumberOfTransitions());
		System.out.println("# emission: " + model.getNumberOfEmissions());

		if (testset == null)
			return;

		model.applyLog();

		// Feature labels.
		String observationFtrLabel = trainset
				.getFeatureLabel(observationFeature);
		String stateFtrLabel = trainset.getFeatureLabel(stateFeature);

		// Test the model on a testset.
		model.setEmissionSmoothingProbability(1e-6);
		model.normalizeProbabilities();
		model.applyLog();
		model.tag(testset, observationFtrLabel, "ne");
		model.setEmissionSmoothingProbability(0.0);
		model.normalizeProbabilities();
		model.applyLog();

		// Evaluate the predicted values.
		Evaluation ev = new Evaluation("0");
		Map<String, Performance> results = ev.evaluateSequences(testset,
				stateFtrLabel, "ne");

		String[] labelOrder = { "LOC", "MISC", "ORG", "PER", "overall" };

		// Write precision, recall and F-1 values.
		System.out.println("Iteration #" + iter);
		System.out.println("|  *Class*  |  *P*  |  *R*  |  *F*  |");
		for (String label : labelOrder) {
			Performance res = results.get(label);
			if (res == null)
				continue;
			System.out.println(String.format(
					"|  %s  |  %6.2f |  %6.2f |  %6.2f |", label,
					100 * res.getPrecision(), 100 * res.getRecall(),
					100 * res.getF1()));
		}
	}

	/**
	 * Modified forward algorithm. For all tagged tokens, it does not consider
	 * any other state possibility but the correct tagged state.
	 */
	protected void forward(DatasetExample example) {
		int size = example.size();

		// Initialize the alpha[0] values.
		Vector<Double> first = alpha.get(0);
		if (tagged[0]) {
			// Set all state probabilities to zero.
			for (int state = 0; state < numStates; ++state)
				first.set(state, 0.0);

			// Get the correct state.
			int taggedState = example.getFeatureValue(0, stateFeature);
			taggedState = model.featureToState.get(taggedState);

			// Set its probability.
			int emission = example.getFeatureValue(0, observationFeature);
			double emisProb = model.getEmissionProbability(emission,
					taggedState)
					* model.getInitialStateProbability(taggedState);

			first.set(taggedState, emisProb);
		} else {
			int emission = example.getFeatureValue(0, observationFeature);
			for (int state = 0; state < numStates; ++state) {
				double startProb = model
						.getEmissionProbability(emission, state)
						* model.getInitialStateProbability(state);
				first.set(state, startProb);
			}
		}

		for (int token = 1; token < size; ++token) {
			// Get the emission symbol in this token.
			int emission = example.getFeatureValue(token, observationFeature);

			if (tagged[token]) {
				// Set the probabilities of every state to zero.
				for (int stateTo = 0; stateTo < numStates; ++stateTo)
					alpha.get(token).set(stateTo, 0.0);

				// Get the correct state.
				int taggedState = example.getFeatureValue(token, stateFeature);
				taggedState = model.featureToState.get(taggedState);

				// Probability of arriving at this state (without emission).
				double prevTokenProb = 0.0;
				for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
					double transProb = model.getTransitionProbability(
							stateFrom, taggedState);
					prevTokenProb += transProb
							* alpha.get(token - 1).get(stateFrom);
				}

				// Complete probability of arriving in this state and emits
				// the current symbol.
				double emisProb = model.getEmissionProbability(emission,
						taggedState);
				alpha.get(token).set(taggedState, emisProb * prevTokenProb);
			} else {
				for (int stateTo = 0; stateTo < numStates; ++stateTo) {
					// Probability of arriving at this state (without emission).
					double prevTokenProb = 0.0;
					for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
						double transProb = model.getTransitionProbability(
								stateFrom, stateTo);
						prevTokenProb += transProb
								* alpha.get(token - 1).get(stateFrom);
					}

					// Complete probability of arriving in this state and emits
					// the current symbol.
					double emisProb = model.getEmissionProbability(emission,
							stateTo);
					alpha.get(token).set(stateTo, emisProb * prevTokenProb);
				}
			}
		}
	}

	/**
	 * Modified backward algorithm. As in the forward algorithm for all tagged
	 * tokens, it does not consider any other state possibility but the correct
	 * tagged state.
	 */
	protected void backward(DatasetExample example) {
		int size = example.size();

		// Initialize the beta[size-1] values.
		Vector<Double> last = beta.get(size - 1);
		if (tagged[size - 1]) {
			// Zero to every state.
			for (int state = 0; state < numStates; ++state)
				last.set(state, 0.0);

			// Get the correct state.
			int taggedState = example.getFeatureValue(size - 1, stateFeature);
			taggedState = model.featureToState.get(taggedState);

			// Set the correct state probability to one.
			last.set(taggedState, 1.0);
		} else {
			for (int state = 0; state < numStates; ++state)
				last.set(state, 1.0);
		}

		for (int token = size - 2; token >= 0; --token) {
			if (tagged[token]) {
				// Set the probability of every state to zero.
				for (int stateFrom = 0; stateFrom < numStates; ++stateFrom)
					beta.get(token).set(stateFrom, 0.0);

				// Get the correct state.
				int taggedState = example.getFeatureValue(token, stateFeature);
				taggedState = model.featureToState.get(taggedState);

				double betaVal = 0.0;
				int emission = example.getFeatureValue(token + 1,
						observationFeature);

				for (int stateTo = 0; stateTo < numStates; ++stateTo) {
					// Probability of emiting the next symbol.
					double emisProb = model.getEmissionProbability(emission,
							stateTo);

					// Probability of transition.
					double transProb = model.getTransitionProbability(
							taggedState, stateTo);

					betaVal += beta.get(token + 1).get(stateTo) * emisProb
							* transProb;
				}

				// Set the value for the correct state.
				beta.get(token).set(taggedState, betaVal);
			} else {
				for (int stateFrom = 0; stateFrom < numStates; ++stateFrom) {
					double betaVal = 0.0;
					int emission = example.getFeatureValue(token + 1,
							observationFeature);

					for (int stateTo = 0; stateTo < numStates; ++stateTo) {
						// Probability of emiting the next symbol.
						double emisProb = model.getEmissionProbability(
								emission, stateTo);

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
	}

	/**
	 * Besides the ordinary temporary variables, also allocate an array of
	 * booleans to store the tagged-flags for the given example.
	 */
	protected void adjustTempVariablesSizes(DatasetExample example) {
		// Do all the ordinary job.
		super.adjustTempVariablesSizes(example);

		// Organize the tagged flags for the given example so that we have a
		// boolean array with a flag to each token.
		int size = example.size();
		if (tagged == null || tagged.length < size)
			tagged = new boolean[size];
		if (taggedExampleFlags != null) {
			Object obj = taggedExampleFlags.get(example.getIndex());
			if (obj instanceof Boolean) {
				boolean flag = (Boolean) obj;
				for (int tkn = 0; tkn < size; ++tkn)
					tagged[tkn] = flag;
			} else if (obj instanceof Vector<?>) {
				@SuppressWarnings("unchecked")
				Vector<? extends Boolean> flags = (Vector<? extends Boolean>) taggedExampleFlags
						.get(example.getIndex());
				for (int tkn = 0; tkn < size; ++tkn)
					tagged[tkn] = flags.get(tkn);
			}
		}
	}
}
