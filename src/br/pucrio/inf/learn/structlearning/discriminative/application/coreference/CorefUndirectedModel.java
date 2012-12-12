package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;

/**
 * Undirected coreference model. This model is based on undirected coreference
 * trees. The prediction problem is solved by Kruskal algorithm. Since the
 * output structures are represented by the same classes (CorefOutput), which
 * supports only directed trees, the update method must be overwritten to
 * disconsider edge orientation.
 * 
 * @author eraldo
 * 
 */
public class CorefUndirectedModel extends CorefModel {

	/**
	 * Logging object.
	 */
	private static Log LOG = LogFactory.getLog(CorefUndirectedModel.class);

	public CorefUndirectedModel(DPTemplateEvolutionModel other)
			throws CloneNotSupportedException {
		super(other);
	}

	public CorefUndirectedModel(int root) {
		super(root);
	}

	@Override
	protected double updateTree(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		/*
		 * The root token must always be ignored during the inference, thus it
		 * has to be always correctly classified.
		 */
		assert outputCorrect.getHead(root) == outputPredicted.getHead(root);

		// Per-token loss value for this example.
		double loss = 0d;
		for (int idxTkn = 0; idxTkn < input.getNumberOfTokens(); ++idxTkn) {

			/*
			 * Check whether the correct edge incoming into mention idxTkn is
			 * present in the predicted output structure (false negatives).
			 */
			int correctHeadMention = outputCorrect.getHead(idxTkn);
			if (correctHeadMention != -1
					&& !isEdgePresent(idxTkn, correctHeadMention,
							outputPredicted)) {
				/*
				 * Sort edge endpoints. In that way, input structures MUST obey
				 * this order to specify edge features, i.e., left mention index
				 * is lower than right mention index.
				 */
				int leftMention = correctHeadMention;
				int rightMention = idxTkn;
				if (leftMention > rightMention) {
					int aux = leftMention;
					leftMention = rightMention;
					rightMention = aux;
				}

				/*
				 * Increment feature parameter weights for correct edge that is
				 * not present in the predicted output.
				 */
				int[] correctFeatures = input.getFeatures(leftMention,
						rightMention);
				if (correctFeatures != null)
					for (int idxFtr = 0; idxFtr < correctFeatures.length; ++idxFtr)
						updateFeatureParam(correctFeatures[idxFtr],
								learningRate);
				else
					LOG.warn("Inexistent edge in correct structure.");

				// Increment (per-token) loss value.
				loss += 0.5d;
			}

			/*
			 * Check whether the predicted edge incoming into mention idxTkn is
			 * present in the correct output structure (false positives).
			 */
			int predictedHeadMention = outputPredicted.getHead(idxTkn);
			if (predictedHeadMention != -1
					&& !isEdgePresent(idxTkn, predictedHeadMention,
							outputCorrect)) {
				/*
				 * Sort edge endpoints. In that way, input structures MUST obey
				 * this order to specify edge features, i.e., left mention index
				 * is lower than right mention index.
				 */
				int leftMention = predictedHeadMention;
				int rightMention = idxTkn;
				if (leftMention > rightMention) {
					int aux = leftMention;
					leftMention = rightMention;
					rightMention = aux;
				}

				/*
				 * Decrement parameter weights for incorrectly predicted edge
				 * features.
				 */
				int[] predictedFeatures = input.getFeatures(leftMention,
						rightMention);
				if (predictedFeatures != null)
					for (int idxFtr = 0; idxFtr < predictedFeatures.length; ++idxFtr)
						updateFeatureParam(predictedFeatures[idxFtr],
								-learningRate);
				else
					LOG.warn("Inexistent edge in correct structure.");

				// Increment (per-token) loss value.
				loss += 0.5d;
			}
		}

		return loss;
	}

	/**
	 * Check whether the given edge (m1,m2) is present in the given output
	 * structure. This method checks both orientations of the given edge.
	 * 
	 * @param m1
	 * @param m2
	 * @param output
	 * @return
	 */
	protected boolean isEdgePresent(int m1, int m2, CorefOutput output) {
		int m2Output = output.getHead(m1);
		if (m2Output == m2)
			// Same orientation.
			return true;

		int m1Output = output.getHead(m2);
		if (m1Output == m1)
			// Inverted orientation.
			return true;

		// Not present.
		return false;
	}

	@Override
	protected double updateAll(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		throw new NotImplementedException();
	}

	@Override
	protected double updateCluster(CorefInput input, CorefOutput outputCorrect,
			CorefOutput outputPredicted, double learningRate) {
		throw new NotImplementedException();
	}

}
