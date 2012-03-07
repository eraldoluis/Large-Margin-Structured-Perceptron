package br.pucrio.inf.learn.structlearning.discriminative.application.coreference.data;

import java.io.FileNotFoundException;
import java.io.IOException;

import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;

/**
 * Represent a depedency parsing dataset.
 * 
 * @author eraldo
 * 
 */
public interface DPDataset extends Dataset {

	@Override
	public CRInput[] getInputs();

	@Override
	public DPOutput[] getOutputs();

	/**
	 * Return the number of tokens in the longest sentence within this dataset.
	 * 
	 * @return
	 */
	public int getMaxNumberOfTokens();

	/**
	 * Return the feature encoding of this dataset.
	 * 
	 * @return
	 */
	public FeatureEncoding<String> getFeatureEncoding();

	/**
	 * Write the content of this dataset in the given file using Java
	 * Serialization API, i.e., using a binary format.
	 * 
	 * @param filename
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void serialize(String filename) throws FileNotFoundException,
			IOException;

	/**
	 * Load examples from input file and imediately serialize them without
	 * storing them.
	 * 
	 * @param inFilename
	 * @param outFilename
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void serialize(String inFilename, String outFilename)
			throws IOException, DatasetException;

	/**
	 * Read the content of this object from the given file. The content of the
	 * file must has been generated by the method <code>serialize()</code> of
	 * this class.
	 * 
	 * @param filename
	 * @throws IOException
	 * @throws FileNotFoundException
	 * @throws ClassNotFoundException
	 */
	public void deserialize(String filename) throws FileNotFoundException,
			IOException, ClassNotFoundException;

}