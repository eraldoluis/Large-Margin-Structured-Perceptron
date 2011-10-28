package br.pucrio.inf.learn.structlearning.discriminative.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Represent a sequence of examples. Each example comprises at least an input
 * structure (<code>ExampleInput</code>) but can also include a corresponding
 * output structure (<code>ExampleOutput</code>) when the dataset is used for
 * train or evaluation.
 * 
 * @author eraldo
 * 
 */
public interface Dataset {

	/**
	 * @return the array of input structures
	 */
	public ExampleInput[] getInputs();

	/**
	 * @return the array of output structures
	 */
	public ExampleOutput[] getOutputs();

	/**
	 * Return the input structure of the example at the given index.
	 * 
	 * @param index
	 * @return
	 */
	public ExampleInput getInput(int index);

	/**
	 * Return the output structure of the example at the given index.
	 * 
	 * @param index
	 * @return
	 */
	public ExampleOutput getOutput(int index);

	/**
	 * @return the number of examples in this dataset.
	 */
	public int getNumberOfExamples();

	/**
	 * @return <code>true</code> if it is training dataset
	 */
	public boolean isTraining();

	/**
	 * Fill this dataset from the given file.
	 * 
	 * @param fileName
	 */
	public void load(String fileName) throws IOException, DatasetException;

	public void load(BufferedReader reader) throws IOException,
			DatasetException;

	public void load(InputStream is) throws IOException, DatasetException;

	/**
	 * Save this dataset in the given file.
	 * 
	 * @param fileName
	 */
	public void save(String fileName) throws IOException, DatasetException;

	public void save(BufferedWriter writer) throws IOException,
			DatasetException;

	public void save(OutputStream os) throws IOException, DatasetException;

}
