package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;

public class CorefModel extends DPTemplateEvolutionModel {

	/**
	 * Create an empty coreference model.
	 * 
	 * @param root
	 */
	public CorefModel(int root) {
		super(root);
	}

	/**
	 * Create a copy of the given model.
	 * 
	 * @param other
	 * @throws CloneNotSupportedException
	 */
	public CorefModel(DPTemplateEvolutionModel other)
			throws CloneNotSupportedException {
		super(other);
	}

	@Override
	protected double update(DPInput input, DPOutput outputCorrect,
			DPOutput outputPredicted, double learningRate) {
		return update((CorefInput) input, (CorefOutput) outputCorrect,
				(CorefOutput) outputPredicted, learningRate);
	}

	protected double update(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		/*
		 * The root node must always be ignored during prediction, thus it has
		 * to be always correctly classified.
		 */
		assert outputCorrect.getHead(root) == outputPredicted.getHead(root);

		double loss = 0d;
		for (int rightMention = 0; rightMention < input.getNumberOfTokens(); ++rightMention) {
			// Skip root node.
			if (rightMention == root)
				continue;

			boolean error = false;

			// Predicted left mention.
			int predictedLeftMention = outputPredicted.getHead(rightMention);
			// Correct left mention.
			int correctLeftMention = outputCorrect.getHead(rightMention);

			if (predictedLeftMention == root) {
				if (correctLeftMention != root)
					error = true;
			} else {
				// Clusters ids of the predicted edge end points.
				int correctClusterOfRightMention = outputCorrect
						.getClusterId(rightMention);
				int correctClusterOfPredictedLeftMention = outputCorrect
						.getClusterId(predictedLeftMention);

				/*
				 * Update model if the predicted edge is incorrect
				 * (intercluster).
				 */
				if (correctClusterOfRightMention != correctClusterOfPredictedLeftMention)
					error = true;
			}

			// Update model.
			if (error) {
				// Decrement weight of the predicted intercluster edge.
				updateFeatures(
						input.getFeatures(predictedLeftMention, rightMention),
						-learningRate);

				// Increment weight of the correct intracluster edge.
				updateFeatures(
						input.getFeatures(correctLeftMention, rightMention),
						learningRate);

				// Increment (per-token) loss value.
				loss += 1d;
			}

		}

		return loss;
	}

	/**
	 * Update the given array of feature codes with the given value.
	 * 
	 * @param features
	 * @param value
	 */
	protected void updateFeatures(int[] features, double value) {
		if (features == null)
			return;
		for (int idxFtr = 0; idxFtr < features.length; ++idxFtr)
			updateFeatureParam(features[idxFtr], value);
	}

	@Override
	public CorefModel clone() throws CloneNotSupportedException {
		return new CorefModel(this);
	}

}
