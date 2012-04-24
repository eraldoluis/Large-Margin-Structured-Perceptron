package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;

public class CorefModel extends DPTemplateEvolutionModel {

	/**
	 * If this flag is <code>true</code>, then update all (correct) edges that
	 * connect two mentions that are not connected in the predicted clustering.
	 */
	private boolean updateAllFalseNegatives;

	/**
	 * Create an empty coreference model.
	 * 
	 * @param root
	 */
	public CorefModel(int root) {
		super(root);
		this.updateAllFalseNegatives = false;
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
		this.updateAllFalseNegatives = false;
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

		if (updateAllFalseNegatives) {
			// Update all false positive and all false negative edges.
			for (int rightMention = 0; rightMention < input.getNumberOfTokens(); ++rightMention) {
				int correctClusterOfRightMention = outputCorrect
						.getClusterId(rightMention);
				int predictedClusterOfRightMention = outputPredicted
						.getClusterId(rightMention);

				// Correct and predicted left mentions in the latent structure.
				int correctLeftMention = outputCorrect.getHead(rightMention);
				int predictedLeftMention = outputPredicted
						.getHead(rightMention);
				if (correctLeftMention != predictedLeftMention) {
					if (predictedLeftMention == root)
						// Decrement incorrectly predicted root.
						updateFeatures(input.getFeatures(predictedLeftMention,
								rightMention), -learningRate);
					else if (correctLeftMention == root)
						// Increment correct root from latent structure.
						updateFeatures(input.getFeatures(correctLeftMention,
								rightMention), learningRate);
				}

				for (int leftMention = 0; leftMention < rightMention; ++leftMention) {
					// Skip artificial root mention.
					if (leftMention == root)
						continue;

					int correctClusterOfLeftMention = outputCorrect
							.getClusterId(leftMention);
					int predictedClusterOfLeftMention = outputPredicted
							.getClusterId(leftMention);
					if (correctClusterOfLeftMention != correctClusterOfRightMention) {
						if (predictedClusterOfLeftMention == predictedClusterOfRightMention)
							/*
							 * Different cluster mentions put together in the
							 * same cluster (false positive).
							 */
							updateFeatures(input.getFeatures(leftMention,
									rightMention), -learningRate);
					} else {
						if (predictedClusterOfLeftMention != predictedClusterOfRightMention)
							/*
							 * Same cluster mentions put in different clusters
							 * (false negative).
							 */
							updateFeatures(input.getFeatures(leftMention,
									rightMention), learningRate);
					}
				}
			}
		} else {
			for (int rightMention = 0; rightMention < input.getNumberOfTokens(); ++rightMention) {
				// Skip root node.
				if (rightMention == root)
					continue;

				boolean error = false;

				// Predicted left mention.
				int predictedLeftMention = outputPredicted
						.getHead(rightMention);
				// Correct left mention.
				int correctLeftMention = outputCorrect.getHead(rightMention);

				if (predictedLeftMention == root) {
					/*
					 * For edges from the root mention, use the latent structure
					 * as reference. That is, if the right mention has been
					 * connected to the root mention by the partial inference
					 * algorithm and the right mention has not been connected to
					 * the root mention by the *complete* inference algorithm,
					 * then update the model.
					 */
					if (correctLeftMention != root)
						error = true;
				} else {
					/*
					 * For ordinary edges (connecting two real mentions), use
					 * the clustering definitions.
					 */
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
					updateFeatures(input.getFeatures(predictedLeftMention,
							rightMention), -learningRate);

					// Increment weight of the correct intracluster edge.
					updateFeatures(
							input.getFeatures(correctLeftMention, rightMention),
							learningRate);

					// Increment (per-token) loss value.
					loss += 1d;
				}

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

	/**
	 * If this value is set to <code>true</code>, then the update procedure will
	 * update every false negative edges given the correct clustering, instead
	 * of updating only the reference correct edge.
	 * 
	 * @param flag
	 */
	public void setUpdateAllFalseNegatives(boolean flag) {
		this.updateAllFalseNegatives = flag;
	}

}
