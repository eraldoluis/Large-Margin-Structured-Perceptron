package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;

/**
 * Represent a dependecy parsing model (head-dependent edge parameters) by means
 * of a set of templates that conjoing basic features within the input
 * structure.
 * 
 * In this version, templates are partitioned and each partition is used once at
 * a time. For each partition, some learning iterations are performed
 * considering only the features from this template partition. Then, the current
 * weights for these features are fixed and the corresponding accumulated
 * weights for each edge is stored for efficiency matter and the next template
 * partition is used for the next learning iterations.
 * 
 * 
 * @author eraldo
 * 
 */
public class DPTemplateEvolutionModel implements DPModel {

	/**
	 * Weight for each feature code (model parameters).
	 */
	private Map<Integer, AveragedParameter> parameters;

	/**
	 * Set of parameters that have been updated in the current iteration.
	 */
	private Set<AveragedParameter> updatedParameters;

	/**
	 * Create a new model with the given template partitions.
	 */
	public DPTemplateEvolutionModel() {
		parameters = new HashMap<Integer, AveragedParameter>();
		updatedParameters = new HashSet<AveragedParameter>();
	}

	/**
	 * Copy constructor.
	 * 
	 * @param other
	 * @throws CloneNotSupportedException
	 */
	@SuppressWarnings("unchecked")
	protected DPTemplateEvolutionModel(DPTemplateEvolutionModel other)
			throws CloneNotSupportedException {
		// Shallow-copy parameters map.
		this.parameters = (HashMap<Integer, AveragedParameter>) ((HashMap<Integer, AveragedParameter>) other.parameters)
				.clone();
		// Clone each map value.
		for (Entry<Integer, AveragedParameter> entry : parameters.entrySet())
			entry.setValue(entry.getValue().clone());

		// Updated parameters and features are NOT copied.
		updatedParameters = new TreeSet<AveragedParameter>();
	}

	/**
	 * Return an edge weight based only on the current features in
	 * <code>activeFeatures</code> list.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDependent
	 * @return
	 */
	protected double getEdgeScoreFromCurrentFeatures(DPInput input,
			int idxHead, int idxDependent) {
		// Get list of feature codes in the given edge.
		int[] features = input.getFeatures(idxHead, idxDependent);

		// Check edge existence.
		if (features == null)
			return Double.NaN;

		double score = 0d;
		for (int idxFtr = 0; idxFtr < features.length; ++idxFtr) {
			AveragedParameter param = parameters.get(features[idxFtr]);
			if (param != null)
				score += param.get();
		}

		return score;
	}

	@Override
	public double getEdgeScore(DPInput input, int idxHead, int idxDependent) {
		// int idxEx = input.getTrainingIndex();
		double score = getEdgeScoreFromCurrentFeatures(input, idxHead,
				idxDependent);
		return score;
		// TODO return fixedWeights[idxEx][idxHead][idxDependent] + score;
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((DPInput) input, (DPOutput) outputCorrect,
				(DPOutput) outputPredicted, learningRate);
	}

	/**
	 * Update this model using the differences between the correct output and
	 * the predicted output, both given as arguments.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 * @return
	 */
	private double update(DPInput input, DPOutput outputCorrect,
			DPOutput outputPredicted, double learningRate) {
		/*
		 * The root token (zero) must always be ignored during the inference,
		 * thus it has to be always correctly classified.
		 */
		assert outputCorrect.getHead(0) == outputPredicted.getHead(0);

		// Per-token loss value for this example.
		double loss = 0d;
		for (int idxTkn = 1; idxTkn < input.getNumberOfTokens(); ++idxTkn) {
			// Correct head token.
			int idxCorrectHead = outputCorrect.getHead(idxTkn);

			// Predicted head token.
			int idxPredictedHead = outputPredicted.getHead(idxTkn);

			// Skip. Correctly predicted head.
			if (idxCorrectHead == idxPredictedHead)
				continue;

			if (idxCorrectHead == -1)
				/*
				 * Skip tokens with missing CORRECT edge (this is due to prune
				 * preprocessing).
				 */
				continue;

			/*
			 * Misclassified head for this token. Thus, update edges parameters.
			 */

			// Increment parameter weights for correct edge features.
			int[] correctFeatures = input.getFeatures(idxCorrectHead,
					idxTkn);
			if (correctFeatures != null)
				for (int idxFtr = 0; idxFtr < correctFeatures.length; ++idxFtr)
					updateFeatureParam(correctFeatures[idxFtr], learningRate);

			if (idxPredictedHead == -1)
				continue;

			/*
			 * Decrement parameter weights for incorrectly predicted edge
			 * features.
			 */
			int[] predictedFeatures = input.getFeatures(idxPredictedHead,
					idxTkn);
			if (predictedFeatures != null)
				for (int idxFtr = 0; idxFtr < predictedFeatures.length; ++idxFtr)
					updateFeatureParam(predictedFeatures[idxFtr], -learningRate);

			// Increment (per-token) loss value.
			loss += 1d;
		}

		return loss;
	}

	/**
	 * Recover the parameter associated with the given feature.
	 * 
	 * If the parameter has not been initialized yet, then create it. If the
	 * inverted index is activated and the parameter has not been initialized
	 * yet, then update the active features lists for each edge where the
	 * feature occurs.
	 * 
	 * @param ftr
	 * @param value
	 * @return
	 */
	private void updateFeatureParam(int code, double value) {
		AveragedParameter param = parameters.get(code);
		if (param == null) {
			// Create a new parameter.
			param = new AveragedParameter();
			parameters.put(code, param);
		}

		// Update parameter value.
		param.update(value);

		// Keep track of updated parameter within this example.
		updatedParameters.add(param);
	}

	@Override
	public void sumUpdates(int iteration) {
		for (AveragedParameter parm : updatedParameters)
			parm.sum(iteration);
		updatedParameters.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		for (AveragedParameter parm : parameters.values())
			parm.average(numberOfIterations);
	}

	@Override
	public DPTemplateEvolutionModel clone() throws CloneNotSupportedException {
		return new DPTemplateEvolutionModel(this);
	}

	@Override
	public void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		throw new NotImplementedException();
	}

	@Override
	public int getNonZeroParameters() {
		return parameters.size();
	}

}
