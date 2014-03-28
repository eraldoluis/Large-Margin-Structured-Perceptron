package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

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
import br.pucrio.inf.learn.util.maxbranching.SimpleWeightedEdge;

/**
 * Rank model. Just an array of weights (one for each feature).
 * 
 * @author eraldo
 * 
 */
public class BisectionModel implements Model {

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
	public BisectionModel() {
		parameters = new HashMap<Integer, AveragedParameter>();
		updatedParameters = new HashSet<AveragedParameter>();
	}

	/**
	 * Create a new model using the given array of parameters weights.
	 * 
	 * @param parameters
	 * @throws CloneNotSupportedException
	 */
	protected BisectionModel(BisectionModel other)
			throws CloneNotSupportedException {
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
	public double update(BisectionInput input, BisectionOutput outputCorrect,
			BisectionOutput outputPredicted, double learningRate) {
		// Get correct and predicted MSTs.
		Set<SimpleWeightedEdge> mstCorrect = outputCorrect.getMst();
		Set<SimpleWeightedEdge> mstPredicted = outputPredicted.getMst();

		// Compute missing edges in the predicted MST.
		Set<SimpleWeightedEdge> missing = new HashSet<SimpleWeightedEdge>(
				mstCorrect);
		missing.removeAll(mstPredicted);

		// Compute mispredicted edges in the predicted MST.
		Set<SimpleWeightedEdge> mispredicted = new HashSet<SimpleWeightedEdge>(
				mstPredicted);
		mispredicted.removeAll(mstCorrect);

		// Loss.
		double loss = 0d;

		// Increment weights of features within missing edges.
		for (SimpleWeightedEdge edge : missing) {
			updateParameters(input.getFeatureCodes(edge.from, edge.to),
					input.getFeatureValues(edge.from, edge.to), learningRate);
			loss += 0.5;
		}

		// Decrement weights of features within mispredicted edges.
		for (SimpleWeightedEdge edge : mispredicted) {
			updateParameters(input.getFeatureCodes(edge.from, edge.to),
					input.getFeatureValues(edge.from, edge.to), -learningRate);
			loss += 0.5;
		}

		return loss;
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((BisectionInput) input, (BisectionOutput) outputCorrect,
				(BisectionOutput) outputPredicted, learningRate);
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
	public BisectionModel clone() throws CloneNotSupportedException {
		return new BisectionModel(this);
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
	 * Update the weights of the given features (codes) by summing the given
	 * values multiplied by the given step.
	 * 
	 * @param codes
	 * @param values
	 * @param step
	 */
	public void updateParameters(int[] codes, double[] values, double step) {
		int numFtrs = codes.length;
		for (int idx = 0; idx < numFtrs; ++idx)
			updateFeatureParam(codes[idx], values[idx] * step);
	}

	public int getNumberOfUpdatedParameters() {
		return parameters.size();
	}
}
