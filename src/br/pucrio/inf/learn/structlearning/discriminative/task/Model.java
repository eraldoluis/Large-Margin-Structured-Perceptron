package br.pucrio.inf.learn.structlearning.discriminative.task;

import java.io.PrintStream;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;

/**
 * Interface of a task-specific model.
 * 
 * @author eraldof
 * 
 */
public interface Model {

	/**
	 * Update this model according to the two outputs (correct and predicted)
	 * for the given input.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 * @return the loss between the correct and the predicted outputs.
	 */
	double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate);

	/**
	 * Account the updates done during the last iteration.
	 * 
	 * @param iteration
	 */
	void sumUpdates(int iteration);

	/**
	 * Average the parameters of all iterations.
	 * 
	 * @param numberOfIterations
	 */
	void average(int numberOfIterations);

	/**
	 * Serialize the model to the given stream.
	 * 
	 * @param ps
	 * @param featureEncoding
	 * @param stateEncoding
	 */
	void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding);

	/**
	 * Return an identical copy of this object.
	 * 
	 * @return
	 * @throws CloneNotSupportedException
	 */
	Model clone() throws CloneNotSupportedException;

}
