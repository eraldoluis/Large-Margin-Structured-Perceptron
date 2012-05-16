package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Store the weight of each feature (indexed by feature code).
 * 
 * This is a model based only on explicit features. That is, inputs must
 * explicitly list all their features. The class <code>DPTemplateModel</code> is
 * more efficient for problems with huge number of features because it does not
 * require as input the explicit list of compound features.
 * 
 * @author eraldo
 * 
 */
public class DPBasicModel implements DPModel {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(DPBasicModel.class);

	/**
	 * Weight for each feature code.
	 */
	private HashMap<Integer, AveragedParameter> featureWeights;

	/**
	 * Set of parameters that have been updated in the current iteration.
	 */
	private Set<AveragedParameter> updatedWeights;

	/**
	 * Create a model with the given total number of features.
	 * 
	 */
	public DPBasicModel() {
		featureWeights = new HashMap<Integer, AveragedParameter>();
		updatedWeights = new HashSet<AveragedParameter>();
	}

	/**
	 * Copy constructor.
	 * 
	 * @param featureWeights
	 */
	@SuppressWarnings("unchecked")
	protected DPBasicModel(HashMap<Integer, AveragedParameter> featureWeights) {
		this.featureWeights = (HashMap<Integer, AveragedParameter>) featureWeights
				.clone();
		for (Entry<Integer, AveragedParameter> entry : this.featureWeights
				.entrySet()) {
			try {
				entry.setValue(entry.getValue().clone());
			} catch (CloneNotSupportedException e) {
				// This should never happen.
				LOG.error("Cloning DP basic model", e);
			}
		}
	}

	/**
	 * Return the parameter associated with the given feature code or
	 * <code>null</code> if such parameter does not exist.
	 * 
	 * @param ftr
	 * @return
	 */
	public AveragedParameter getFeatureWeight(int ftr) {
		return featureWeights.get(ftr);
	}

	/**
	 * Return the parameter object associated with the given feature code. If
	 * such parameter does not exist, then create a new one, put it in the
	 * parameter map and return it.
	 * 
	 * @param ftr
	 * @return
	 */
	protected AveragedParameter getFeatureWeightOrCreate(int ftr) {
		AveragedParameter param = featureWeights.get(ftr);
		if (param == null) {
			param = new AveragedParameter();
			featureWeights.put(ftr, param);
		}
		return param;
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
		 * The root token (zero) must always be ignored during the inference and
		 * so its always correctly classified (its head must always be pointing
		 * to itself).
		 */
		assert outputCorrect.getHead(0) == outputPredicted.getHead(0);

		// Per-token loss value for this example.
		double loss = 0d;
		for (int idxTkn = 1; idxTkn < input.getNumberOfTokens(); ++idxTkn) {
			int idxCorrectHead = outputCorrect.getHead(idxTkn);
			int idxPredictedHead = outputPredicted.getHead(idxTkn);
			if (idxCorrectHead == idxPredictedHead)
				// Correctly predicted head.
				continue;

			if (idxCorrectHead == -1)
				/*
				 * Skip tokens with missing CORRECT edge (this is due to prune
				 * preprocessing).
				 */
				continue;

			// Misclassified head. Increment missed edges weights.
			int[] correctFeatures = input.getFeatures(idxCorrectHead, idxTkn);
			for (int idxFtr = 0; idxFtr < correctFeatures.length; ++idxFtr) {
				int ftr = correctFeatures[idxFtr];
				AveragedParameter param = getFeatureWeightOrCreate(ftr);
				param.update(learningRate);
				updatedWeights.add(param);
			}

			if (idxPredictedHead == -1) {
				LOG.warn("Predicted head is -1");
				continue;
			}

			// Decrement mispredicted edges weights.
			int[] predictedFeatures = input.getFeatures(idxPredictedHead,
					idxTkn);
			for (int idxFtr = 0; idxFtr < predictedFeatures.length; ++idxFtr) {
				int ftr = predictedFeatures[idxFtr];
				AveragedParameter param = getFeatureWeightOrCreate(ftr);
				param.update(-learningRate);
				updatedWeights.add(param);
			}

			// Increment (per-token) loss value.
			loss += 1d;
		}

		return loss;
	}

	@Override
	public void sumUpdates(int iteration) {
		for (AveragedParameter parm : updatedWeights)
			parm.sum(iteration);
		updatedWeights.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		for (AveragedParameter parm : featureWeights.values())
			parm.average(numberOfIterations);
	}

	@Override
	public double getEdgeScore(DPInput input, int idxHead, int idxDependent) {
		double score = 0d;
		// Accumulate the parameter in the edge score.
		int[] ftrs = input.getFeatures(idxHead, idxDependent);
		if (ftrs == null)
			// Edge does not exist.
			return Double.NaN;
		for (int ftr : ftrs) {
			AveragedParameter param = getFeatureWeight(ftr);
			if (param != null)
				score += param.get();
		}
		return score;
	}

	@Override
	public void save(String fileName, Dataset dataset) throws IOException,
			FileNotFoundException {
		throw new NotImplementedException();
	}

	@Override
	public DPBasicModel clone() throws CloneNotSupportedException {
		return new DPBasicModel(featureWeights);
	}

	@Override
	public int getNumberOfUpdatedParameters() {
		return featureWeights.size();
	}

}
