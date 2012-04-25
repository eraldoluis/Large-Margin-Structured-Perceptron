package br.pucrio.inf.learn.structlearning.discriminative.task;

import java.io.FileNotFoundException;
import java.io.IOException;

import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

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
	 * Save this model to the given filename.
	 * 
	 * @param fileName
	 *            name of the file where the model is to be saved.
	 * @param dataset
	 *            the dataset where the underlying encodings used by this model
	 *            are. Usually, this is the training dataset used to generate
	 *            this model.
	 */
	public void save(String fileName, Dataset dataset) throws IOException,
			FileNotFoundException;

	/**
	 * Return an identical copy of this object.
	 * 
	 * @return
	 * @throws CloneNotSupportedException
	 */
	public Model clone() throws CloneNotSupportedException;

}
