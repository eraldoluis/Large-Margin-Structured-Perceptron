package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.io.PrintStream;
import java.util.HashSet;
import java.util.Set;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Store the weight of each feature (indexed by feature code).
 * 
 * @author eraldo
 * 
 */
public class DPModel implements Model {

	// TODO debug
	public static double maxAbsParam;
	public static double maxAbsParamSum;

	/**
	 * Weight for each feature code.
	 */
	private AveragedParameter[] featureWeights;

	/**
	 * Set of parameters that have been updated in the current iteration.
	 */
	private Set<AveragedParameter> updatedWeights;

	/**
	 * Create a model with the given total number of features.
	 * 
	 * @param numberOfFeatures
	 */
	public DPModel(int numberOfFeatures) {
		featureWeights = new AveragedParameter[numberOfFeatures];
		for (int idxFtr = 0; idxFtr < numberOfFeatures; ++idxFtr)
			featureWeights[idxFtr] = new AveragedParameter();
		updatedWeights = new HashSet<AveragedParameter>();
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

			// Misclassified head. Increment missed edges weights.
			for (int ftr : input.getFeatureCodes(idxCorrectHead, idxTkn)) {
				featureWeights[ftr].update(learningRate);
				updatedWeights.add(featureWeights[ftr]);
			}

			// Decrement mispredicted edges weights.
			for (int ftr : input.getFeatureCodes(idxPredictedHead, idxTkn)) {
				featureWeights[ftr].update(-learningRate);
				updatedWeights.add(featureWeights[ftr]);
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
		for (AveragedParameter parm : featureWeights) {
			parm.average(numberOfIterations);
			// TODO debug
			if (parm.get() > maxAbsParam)
				maxAbsParam = parm.get();
		}
	}

	/**
	 * Return the weight of the feature with code <code>featureCode</code>.
	 * 
	 * @param featureCode
	 * @return
	 */
	public double getFeatureWeight(int featureCode) {
		return featureWeights[featureCode].get();
	}

	@Override
	public void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		throw new NotImplementedException();
	}

	@Override
	public DPModel clone() throws CloneNotSupportedException {
		DPModel copy = new DPModel(featureWeights.length);
		for (int idxFtr = 0; idxFtr < featureWeights.length; ++idxFtr)
			copy.featureWeights[idxFtr] = featureWeights[idxFtr].clone();
		return copy;
	}

}
