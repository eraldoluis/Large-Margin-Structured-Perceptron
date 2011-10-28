package br.pucrio.inf.learn.structlearning.discriminative.evaluation;

import java.util.Map;
import java.util.TreeMap;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Evaluation method that is based on accuracy of a set of items.
 * 
 * @author eraldo
 * 
 */
public abstract class AccuracyEvaluation {

	/**
	 * Evaluate the given sequence of examples and return, at least, the average
	 * accuracy and the accuracy per example.
	 * 
	 * @param inputs
	 * @param corrects
	 * @param predicteds
	 * @return
	 */
	public Map<String, Double> evaluateExamples(ExampleInput[] inputs,
			ExampleOutput[] corrects, ExampleOutput[] predicteds) {

		// Store the average accuracy and the accucary per-examples.
		TreeMap<String, Double> res = new TreeMap<String, Double>();

		// The average performance.
		double numItems = 0;
		double numCorrectItems = 0;

		// Number of correctly classified examples.
		double numCorrectExamples = 0;

		// Evaluate each sentence.
		for (int idxSeq = 0; idxSeq < inputs.length; ++idxSeq) {
			ExampleInput input = inputs[idxSeq];
			ExampleOutput correct = corrects[idxSeq];
			ExampleOutput predicted = predicteds[idxSeq];

			// Evaluate example.
			int[] correctAndTotal = evaluateExample(input, correct, predicted);
			numItems += correctAndTotal[1];
			numCorrectItems += correctAndTotal[0];

			if (correctAndTotal[0] == correctAndTotal[1])
				// The whole example is correctly predicted.
				numCorrectExamples += 1;
		}

		res.put("example", numCorrectExamples / inputs.length);
		res.put("average", numCorrectItems / numItems);

		return res;
	}

	protected abstract int[] evaluateExample(ExampleInput inputSeq,
			ExampleOutput correctSeq, ExampleOutput predictedSeq);

}
