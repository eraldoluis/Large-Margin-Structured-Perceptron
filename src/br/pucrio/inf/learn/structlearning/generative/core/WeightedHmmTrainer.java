package br.pucrio.inf.learn.structlearning.generative.core;

import java.util.Collection;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;

/**
 * Generative HMM trainer with different weights for the examples.
 * 
 * @author eraldof
 * 
 */
public class WeightedHmmTrainer extends HmmTrainer {

	/**
	 * Logger.
	 */
	private static Log logger = LogFactory.getLog(WeightedHmmTrainer.class);

	/**
	 * Weight for each example. Each weight may be either just a
	 * <code>Double</code>, i.e., the same weight for every token within the
	 * example or a <code>VectorDouble</code> with a weight for each token.
	 */
	protected Vector<Object> weights;

	/**
	 * Default constructor.
	 */
	public WeightedHmmTrainer() {
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

	public HmmModel train(Corpus trainset, String observationFeatureLabel,
			String stateFeatureLabel, String defaultStateLabel)
			throws DatasetException, HmmException {

		if (weights == null)
			throw new HmmException("The example weight vector has null value.");

		this.observationFeature = trainset
				.getFeatureIndex(observationFeatureLabel);
		if (observationFeature < 0)
			throw new DatasetException("Observation feature "
					+ observationFeatureLabel + " does not exist.");

		this.stateFeature = trainset.getFeatureIndex(stateFeatureLabel);
		if (observationFeature < 0)
			throw new DatasetException("State feature " + stateFeatureLabel
					+ " does not exist.");

		Vector<Integer> stateFeatures = new Vector<Integer>();

		// The default state must be the first one in the list.
		stateFeatures.add(trainset.getFeatureValueEncoding().putString(
				defaultStateLabel));

		// Find the set of states within the state feature.
		for (DatasetExample example : trainset) {
			int lenExample = example.size();
			for (int idxTkn = 0; idxTkn < lenExample; ++idxTkn) {
				int ftrVal = example.getFeatureValue(idxTkn, stateFeature);
				if (!stateFeatures.contains(ftrVal))
					stateFeatures.add(ftrVal);
			}
		}

		// Create an empty model. This model is used to accumulate the counters
		// before calculating the corresponding probabilities.
		HmmModel model = new HmmModel(trainset.getFeatureValueEncoding(),
				stateFeatures);

		// Accumulate counters.
		int idxExample = 0;
		for (DatasetExample example : trainset) {
			// Array of toekn weights.
			double[] weight = new double[example.size()];

			// Weight of the current example.
			Object objW = weights.get(idxExample++);

			if (objW instanceof Double) {
				double wForAllTkns = (Double) objW;
				for (int tkn = 0; tkn < weight.length; ++tkn)
					weight[tkn] = (Double) wForAllTkns;
			} else if (objW instanceof Vector<?>) {
				@SuppressWarnings("unchecked")
				Vector<Object> ws = (Vector<Object>) objW;
				for (int tkn = 0; tkn < weight.length; ++tkn)
					weight[tkn] = (Double) ws.get(tkn);
			} else
				throw new HmmException(
						"Example weights must be Double or Vector<Double>.");

			// Example length.
			int lenEx = example.size();
			if (lenEx <= 0) {
				logger.warn("Empty example within training set. Id: "
						+ example.getID());
				continue;
			}

			// First state probability.
			int ftrState = example.getFeatureValue(0, stateFeature);
			model.incProbInitialByFeature(ftrState, weight[0]);

			// Emission probability.
			int ftrObservation = example.getFeatureValue(0, observationFeature);
			model.incProbEmissionByFeature(ftrState, ftrObservation, weight[0]);

			// Remaining states.
			for (int token = 1; token < lenEx; ++token) {
				int stateFrom = example
						.getFeatureValue(token - 1, stateFeature);
				int stateTo = example.getFeatureValue(token, stateFeature);
				ftrObservation = example.getFeatureValue(token,
						observationFeature);
				model.incProbTransitionByFeature(stateFrom, stateTo,
						(weight[token - 1] + weight[token]) / 2);
				model.incProbEmissionByFeature(stateTo, ftrObservation,
						weight[token]);
			}

			// Terminal state.
			ftrState = example.getFeatureValue(lenEx - 1, stateFeature);
			model.incProbFinalByFeature(ftrState, weight[lenEx - 1]);
		}

		// Calculate probabilities by normalizing counters and apply the log.
		// TODO smoothing need the original counters.
		// model.normalizeProbabilities();
		// model.applyLog();

		return model;
	}
}
