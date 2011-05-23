package br.pucrio.inf.learn.structlearning.generative.core;

import java.util.Vector;

import org.apache.log4j.Logger;

import br.pucrio.inf.learn.structlearning.generative.data.Dataset;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;


/**
 * Generative HMM trainer.
 * 
 * @author eraldof
 * 
 */
public class HmmTrainer {

	/**
	 * Logger.
	 */
	private static Logger logger = Logger.getLogger(HmmTrainer.class);

	/**
	 * Index of the feature used as observation value.
	 */
	protected int observationFeature;

	/**
	 * Index of the feature that indicates the state.
	 */
	protected int stateFeature;

	/**
	 * The trainset.
	 */
	protected Dataset trainset;

	/**
	 * Default constructor.
	 */
	public HmmTrainer() {
	}

	public HmmModel train(Dataset trainset, String observationFeatureLabel,
			String stateFeatureLabel, String defaultStateLabel)
			throws DatasetException, HmmException {

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
		for (DatasetExample example : trainset) {
			// Example length.
			int lenEx = example.size();
			if (lenEx <= 0) {
				logger.warn("Empty example within training set. Id: "
						+ example.getID());
				continue;
			}

			// First state.
			int ftrState = example.getFeatureValue(0, stateFeature);
			int ftrObservation = example.getFeatureValue(0, observationFeature);
			model.incProbInitialByFeature(ftrState);
			model.incProbEmissionByFeature(ftrState, ftrObservation);

			// Remaining states.
			for (int token = 1; token < lenEx; ++token) {
				int stateFrom = example
						.getFeatureValue(token - 1, stateFeature);
				int stateTo = example.getFeatureValue(token, stateFeature);
				ftrObservation = example.getFeatureValue(token,
						observationFeature);
				model.incProbTransitionByFeature(stateFrom, stateTo);
				model.incProbEmissionByFeature(stateTo, ftrObservation);
			}

			// Terminal state.
			model.incProbFinalByFeature(example.getFeatureValue(lenEx - 1,
					stateFeature));
		}

		// Calculate probabilities by normalizing counters and take the log.
		// TODO testing for smoothing
		// model.normalizeProbabilities();
		// model.applyLog();

		return model;
	}
}
