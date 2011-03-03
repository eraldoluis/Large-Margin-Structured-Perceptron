package br.pucrio.inf.learn.structlearning.task;

import java.io.PrintStream;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;

/**
 * Interface for a model. One shall implement this interface for any task.
 * 
 * @author eraldof
 * 
 */
public interface Model {

	/**
	 * Fill the output structure using the given input and this model.
	 * 
	 * @param input
	 * @param output
	 */
	public void inference(ExampleInput input, ExampleOutput output);

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
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate);

	/**
	 * Account the updates done during the last iteration.
	 * 
	 * @param iteration
	 */
	public void sumUpdates(int iteration);

	/**
	 * Average the parameters of all iterations.
	 * 
	 * @param numberOfIterations
	 */
	public void average(int numberOfIterations);

	/**
	 * Serialize the model to the given stream.
	 * 
	 * @param ps
	 * @param featureEncoding
	 * @param stateEncoding
	 */
	public void save(PrintStream ps, StringEncoding featureEncoding,
			StringEncoding stateEncoding);

	public Object clone() throws CloneNotSupportedException;

}
