package br.pucrio.inf.learn.structlearning.discriminative.application.rank;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Rank model. Just an array of weights (one for each feature).
 * 
 * @author eraldo
 * 
 */
public class RankModel implements Model {

	/**
	 * Feature parameters.
	 */
	private Map<Integer, AveragedParameter> parameters;

	/**
	 * Store the parameters updated in each training iteration.
	 */
	private Set<AveragedParameter> updatedParameters;

	/**
	 * Create a new empty model.
	 * 
	 */
	public RankModel() {
		parameters = new HashMap<Integer, AveragedParameter>();
		updatedParameters = new HashSet<AveragedParameter>();
	}

	/**
	 * Create a new model using the given array of parameters weights.
	 * 
	 * @param parameters
	 * @throws CloneNotSupportedException
	 */
	protected RankModel(RankModel other) throws CloneNotSupportedException {
		// Shallow-clone parameters map.
		this.parameters = new HashMap<Integer, AveragedParameter>(
				other.parameters);

		// Clone each map value.
		for (Entry<Integer, AveragedParameter> entry : parameters.entrySet())
			entry.setValue(entry.getValue().clone());

		this.updatedParameters = new HashSet<AveragedParameter>();
	}

	/**
	 * Update the parameters of the features that differ from the two given
	 * output rankings.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 * @return the loss between the correct and the predicted output.
	 */
	public double update(RankInput input, RankOutput outputCorrect,
			RankOutput outputPredicted, double learningRate) {
		// Example size (number of queries).
		int size = outputCorrect.size();
		// Sum of precision @ k.
		double avgPrec = 0;
		// Number of irrelevant items @ k.
		int numIrrelevant = 0;
		// Total number of relevant items.
		int numTotalRelevant = outputCorrect.getNumberOfRelevantItems();
		// Iterate over the ordered items of the predicted output.
		for (int idxItem = 0; idxItem < size; ++idxItem) {
			// k to calculate prec@k.
			int k = idxItem + 1;
			// That is the item identifier (index in the input array).
			int item = outputPredicted.getItemAtIndex(idxItem);
			// Item is relevant?
			if (outputCorrect.isRelevant(item)) {
				// Compute precision @ k and add it to the accum variable.
				avgPrec += (k - numIrrelevant) / (double) k;
				/*
				 * Increment parameters weights whenever there are irrevelevant
				 * items before rank k. The update length is proportional to the
				 * number of irrelevant items before rank k.
				 */
				if (numIrrelevant > 0) {
					updateParameters(input.getFeatures(item), learningRate
							* numIrrelevant);
				}
			} else {
				// One more irrelevant item. Hope it's in the bottom ;).
				++numIrrelevant;
				/*
				 * Decrement parameters weights whenever there are relevant
				 * items after rank k. The update length is proportional to the
				 * number of relevant items after rank k.
				 */
				int numRelItemsAfterK = numTotalRelevant - (k - numIrrelevant);
				if (numRelItemsAfterK > 0) {
					updateParameters(input.getFeatures(item), -learningRate
							* numRelItemsAfterK);
				}
			}
		}
		// Average precision.
		avgPrec = avgPrec / (size - numIrrelevant);
		// The loss is equal to what misses to an average precision of 1.
		return avgPrec;
	}

	/**
	 * Calculate the loss of the given predicted output according to the given
	 * correct output.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @return
	 */
	public double loss(RankInput input, RankOutput outputCorrect,
			RankOutput outputPredicted) {
		// Example size (number of queries).
		int size = outputCorrect.size();
		// Sum of precision @ k.
		double avgPrec = 0;
		// Number of irrelevant items @ k.
		int numIrrelevant = 0;
		// Iterate over the ordered items of the predicted output.
		for (int idxItem = 0; idxItem < size; ++idxItem) {
			// k to calculate prec@k.
			int k = idxItem + 1;
			// That is the item identifier (index in the input array).
			int item = outputPredicted.getItemAtIndex(idxItem);
			// Item is relevant?
			if (outputCorrect.isRelevant(item)) {

				// Compute precision @ k and add it to the accum variable.
				avgPrec += (k - numIrrelevant) / (double) k;

			} else {

				// One more irrelevant item. Hope it's in the bottom ;).
				++numIrrelevant;

			}
		}

		// Average precision.
		avgPrec = avgPrec / (size - numIrrelevant);
		// The loss is equal to what misses to an average precision of 1.
		return 1- avgPrec;
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((RankInput) input, (RankOutput) outputCorrect,
				(RankOutput) outputPredicted, learningRate);
	}

	@Override
	public void sumUpdates(int iteration) {
		for (AveragedParameter parm : updatedParameters)
			parm.sum(iteration);
		updatedParameters.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		for (AveragedParameter parm : updatedParameters)
			parm.average(numberOfIterations);
	}

	/**
	 * Return the weight for the given feature code.
	 * 
	 * @param code
	 * @return
	 */
	public double getFeatureWeight(int code) {
		AveragedParameter param = parameters.get(code);
		if (param == null)
			return 0d;
		return param.get();
	}

	@Override
	public RankModel clone() throws CloneNotSupportedException {
		return new RankModel(this);
	}

	@Override
	public void save(String fileName, Dataset dataset) {
		throw new NotImplementedException();
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
	protected void updateFeatureParam(int code, double value) {
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

	/**
	 * Update parameters of the given array of features by adding the given
	 * value.
	 * 
	 * @param ftrs
	 * @param val
	 */
	public void updateParameters(int[] ftrs, double val) {
		for (int ftr : ftrs)
			updateFeatureParam(ftr, val);
	}

	public int getNumberOfUpdatedParameters() {
		return parameters.size();
	}
}
