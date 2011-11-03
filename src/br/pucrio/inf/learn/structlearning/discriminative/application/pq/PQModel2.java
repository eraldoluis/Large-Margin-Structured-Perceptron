package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.io.PrintStream;
import java.util.Iterator;

import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
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
	public double update(PQInput2 input, PQOutput2 outputCorrect,
						PQOutput2 outputPredicted, double learningRate) {
		boolean isDifferent = false;
		int outputCorrectSize = outputCorrect.size();
		
		for (int i = 0; i < outputCorrectSize; ++i) {
			int labelCorrect   = outputCorrect.getAuthor(i);
			int labelPredicted = outputPredicted.getAuthor(i);
			
			if (labelCorrect != labelPredicted) {
				isDifferent = true;
				
				Iterator<Integer> it = input.getFeatureCodes(i, labelCorrect).iterator();
				int featureIndex;
				
				while(it.hasNext()) {
					featureIndex = it.next();
					this.featureWeights[featureIndex] += learningRate;
				}
				
				it = input.getFeatureCodes(i, labelPredicted).iterator();
				while(it.hasNext()) {
					featureIndex = it.next();
					this.featureWeights[featureIndex] -= learningRate;
				}
			}
		}
		
		if (isDifferent)
			return 1d;
		else
			return 0d;
	}
	
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((PQInput2) input, (PQOutput2) outputCorrect,
				(PQOutput2) outputPredicted, learningRate);
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
