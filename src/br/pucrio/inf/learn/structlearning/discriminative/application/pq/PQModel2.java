package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.io.PrintStream;
import java.util.Iterator;

import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Person-quotation model. Just an array of weights (one for each feature).
 * 
 * @author eraldo
 * 
 */
public class PQModel2 implements Model {

	/**
	 * Feature weights.
	 */
	private double[] featureWeights;

	public PQModel2(int numberOfFeatures) {
		featureWeights = new double[numberOfFeatures];
	}

	/**
	 * Update the parameters of the features that differ from the two given
	 * output persons and that are present in the given input sequence.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 * @return the loss between the correct and the predicted output.
	 */
	public double update(PQInput input, PQOutput outputCorrect,
			PQOutput outputPredicted, double learningRate) {
		int labelCorrect = outputCorrect.getPerson();
		int labelPredicted = outputPredicted.getPerson();
		
		if (labelCorrect != labelPredicted) {
			Iterator<Integer> i = input.getFeatureCodes(labelCorrect).iterator();
			int featureIndex;
			
			while(i.hasNext()) {
				featureIndex = i.next();
				this.featureWeights[featureIndex] += learningRate;
			}
			
			i = input.getFeatureCodes(labelPredicted).iterator();
			while(i.hasNext()) {
				featureIndex = i.next();
				this.featureWeights[featureIndex] -= learningRate;
			}
			
			return 1;
		}
		
		return 0;
	}
	
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((PQInput) input, (PQOutput) outputCorrect,
				(PQOutput) outputPredicted, learningRate);
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
	public PQModel2 clone() throws CloneNotSupportedException {
		return null;
	}

	@Override
	public void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		// TODO Auto-generated method stub

	}
	
	public double getFeatureWeight(int featureIndex) {
		return this.featureWeights[featureIndex];
	}

}
