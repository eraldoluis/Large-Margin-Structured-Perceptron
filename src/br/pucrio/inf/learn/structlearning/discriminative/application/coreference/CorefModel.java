package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;

public class CorefModel extends DPTemplateEvolutionModel {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(CorefModel.class);

	/**
	 * Available strategies to update the model.
	 * 
	 * @author eraldo
	 * 
	 */
	public enum UpdateStrategy {
		/**
		 * Update edges based on clusters, that is, only update the features of
		 * an edge if its endpoints mentions are in different clusters but are
		 * predicted in the same cluster or vice-versa. Regarding edges from the
		 * artificial root node (that are not determined by the given correct
		 * clusterings), use the latent edges as correct edges.
		 */
		CLUSTER,

		/**
		 * Update edges based on the latent structure. Even when a predicted
		 * edge does not implicate an incorrect cluster prediction, its weights
		 * are updated if this edge is not present in the latent tree.
		 */
		TREE,

		/**
		 * Update all false negative and all false positive edges based on the
		 * predicted clusterings. Again, as for the <code>CLUSTER</code>
		 * strategy, for edges from the artificial root node (that are not
		 * determined by the given corrrect clusterings), use the latent edges
		 * as correct edges.
		 */
		ALL,
	}

	/**
	 * Indicate which update strategy to use. See <code>UpdateStrategy</code>
	 * enum for available strategies.
	 */
	private UpdateStrategy updateStrategy;

	/**
	 * Create an empty coreference model.
	 * 
	 * @param root
	 */
	public CorefModel(int root) {
		super(root);
		this.updateStrategy = UpdateStrategy.CLUSTER;
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
		this.updateStrategy = UpdateStrategy.CLUSTER;
	}

	@Override
	protected double update(DPInput input, DPOutput outputCorrect,
			DPOutput outputPredicted, double learningRate) {
		return update((CorefInput) input, (CorefOutput) outputCorrect,
				(CorefOutput) outputPredicted, learningRate);
	}

	/**
	 * Update this model based on the differences between the two given output
	 * structures <code>outputCorrect</code> and <code>outputPredicted</code>
	 * for the given input <code>input</code>.
	 * 
	 * @param input
	 *            the input structure.
	 * @param outputCorrect
	 *            the correct output structure for the given input.
	 * @param outputPredicted
	 *            a predicted output structure.
	 * @param learningRate
	 *            the update term that will be summed for the features of
	 *            correct edges and subtracted for the incorrect ones.
	 * @return the loss of the predicted structured, i.e., the difference
	 *         between the predicted and the correct output structure.
	 */
	protected double update(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		/*
		 * The root node must always be ignored during prediction, thus it has
		 * to be always correctly classified.
		 */
		assert outputCorrect.getHead(root) == outputPredicted.getHead(root);

		switch (updateStrategy) {
		case ALL:
			return updateAll(input, outputCorrect, outputPredicted,
					learningRate);
		case TREE:
			return updateTree(input, outputCorrect, outputPredicted,
					learningRate);
		default:
			LOG.warn(String.format(
					"Undefined update strategy (%s). Using CLUSTER.",
					updateStrategy.toString()));
		case CLUSTER:
			return updateCluster(input, outputCorrect, outputPredicted,
					learningRate);
		}
	}

	protected double updateAll(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		double loss = 0d;
		// Update all false positive and all false negative edges.
		for (int rightMention = 0; rightMention < input.getNumberOfTokens(); ++rightMention) {
			int correctClusterOfRightMention = outputCorrect
					.getClusterId(rightMention);
			int predictedClusterOfRightMention = outputPredicted
					.getClusterId(rightMention);

			// Correct and predicted left mentions in the latent structure.
			int correctLeftMention = outputCorrect.getHead(rightMention);
			int predictedLeftMention = outputPredicted.getHead(rightMention);
			if (correctLeftMention != predictedLeftMention) {
				if (predictedLeftMention == root)
					// Decrement incorrectly predicted root.
					updateFeatures(input.getFeatures(predictedLeftMention,
							rightMention), -learningRate);
				else if (correctLeftMention == root)
					// Increment correct root from latent structure.
					updateFeatures(
							input.getFeatures(correctLeftMention, rightMention),
							learningRate);
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
						 * Mentions from different clusters put together in the
						 * same cluster (false positive) by the predicted
						 * structure.
						 */
						updateFeatures(
								input.getFeatures(leftMention, rightMention),
								-learningRate);
				} else {
					if (predictedClusterOfLeftMention != predictedClusterOfRightMention)
						/*
						 * Mentions from the same cluster put in different
						 * clusters (false negative) by the predicted
						 * structured.
						 */
						updateFeatures(
								input.getFeatures(leftMention, rightMention),
								learningRate);
				}
			}
		}
		return loss;
	}

	protected double updateCluster(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
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
				/*
				 * For edges from the root mention, use the latent structure as
				 * reference. That is, if the right mention has been connected
				 * to the root mention by the partial inference algorithm and
				 * the right mention has not been connected to the root mention
				 * by the *complete* inference algorithm, then update the model.
				 */
				if (correctLeftMention != root) {
					int predictedClusterOfCorrectLeftMention = outputPredicted
							.getClusterId(correctLeftMention);
					int predictedClusterOfRightMention = outputPredicted
							.getClusterId(rightMention);
					if (predictedClusterOfCorrectLeftMention != predictedClusterOfRightMention)
						error = true;
					else
						LOG.debug("SKIPPED!!!");
				}
			} else {
				/*
				 * For ordinary edges (connecting two real mentions), use the
				 * clustering definitions.
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

	protected double updateTree(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		// Use the ordinary DP update method that is based only on trees.
		return super
				.update(input, outputCorrect, outputPredicted, learningRate);
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
	 * Set the update strategy for training.
	 * 
	 * @param strategy
	 */
	public void setUpdateStrategy(UpdateStrategy strategy) {
		this.updateStrategy = strategy;
	}
}
