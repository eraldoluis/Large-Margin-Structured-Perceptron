package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.io.PrintStream;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Person-quotation model. Just an array of weights (one for each feature).
 * 
 * @author eraldo
 * 
 */
public class PQModel implements Model {

	/**
	 * Feature weights.
	 */
	private double[] featureWeights;

	public PQModel(int numberOfFeatures) {
		featureWeights = new double[numberOfFeatures];
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void sumUpdates(int iteration) {
		// TODO Auto-generated method stub

	}

	@Override
	public void average(int numberOfIterations) {
		// TODO Auto-generated method stub

	}

	@Override
	public PQModel clone() throws CloneNotSupportedException {
		return null;
	}

	@Override
	public void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		// TODO Auto-generated method stub

	}

}
