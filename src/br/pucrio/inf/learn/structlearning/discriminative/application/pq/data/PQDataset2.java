package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Collection;
import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a text dataset.
 * 
 * Provide methods for manipulating a text dataset. Some operations are: create
 * new features, remove features, get feature values, get a complete example,
 * change feature values.
 * 
 * The feature values are stored as integer. We use a feature-value mapping to
 * encode the string values.
 * 
 */
public class PQDataset2 {

	private static final Log LOG = LogFactory.getLog(PQDataset2.class);

	/**
	 * Indicate whether this is a training dataset or not.
	 */
	protected boolean training;

	/**
	 * Map string feature values to integer values (codes).
	 */
	protected FeatureEncoding<String> featureEncoding;

	/**
	 * Vector of the input-part of the examples.
	 */
	protected PQInput2[] inputExamples;

	/**
	 * Vector of the output-part of the examples (correct predictions).
	 */
	protected PQOutput2[] outputExamples;

	/**
	 * Default constructor.
	 */
	public PQDataset2() {
		this(new StringMapEncoding());
	}

	/**
	 * Create a dataset using the given feature-value encodings. One can use
	 * this constructor to create a dataset compatible with a previous loaded
	 * model, for instance.
	 * 
	 * @param featureEncoding
	 */
	public PQDataset2(FeatureEncoding<String> featureEncoding) {
		this.featureEncoding = featureEncoding;
		this.training = false;
	}

	public PQDataset2(FeatureEncoding<String> featureEncoding, boolean training) {
		this.featureEncoding = featureEncoding;
		this.training = training;
	}

	/**
	 * Load the dataset from a file.
	 * 
	 * @param fileName
	 *            the name of a file to load the dataset.
	 * 
	 * @throws DatasetException
	 * @throws IOException
	 */
	public PQDataset2(String fileName) throws IOException, DatasetException {
		this(new StringMapEncoding());
		load(fileName);
	}

	/**
	 * Load the dataset from a <code>InputStream</code>.
	 * 
	 * @param is
	 * @throws IOException
	 * @throws DatasetException
	 */
	public PQDataset2(InputStream is) throws IOException, DatasetException {
		this(new StringMapEncoding());
		load(is);
	}

	/**
	 * Load the dataset from the given file and use the given feature-value
	 * encodings. One can use this constructor to load a dataset compatible with
	 * a previous loaded model, for instance.
	 * 
	 * @param fileName
	 *            name and path of a file.
	 * @param featureEncoding
	 *            use a determined feature values encoding.
	 * 
	 * @throws IOException
	 *             if occurs some problem reading the file.
	 * @throws DatasetException
	 *             if the file contains invalid data.
	 */
	public PQDataset2(String fileName, FeatureEncoding<String> featureEncoding)
			throws IOException, DatasetException {
		this(featureEncoding);
		load(fileName);
	}

	public PQInput2 getPQInput2(int index) {
		return inputExamples[index];
	}

	/**
	 * Return the number of examples within this dataset.
	 * 
	 * @return the number of examples within this dataset
	 */
	public int getNumberOfExamples() {
		return inputExamples.length;
	}

	public FeatureEncoding<String> getFeatureEncoding() {
		return featureEncoding;
	}

	public PQInput2[] getInputs() {
		return inputExamples;
	}

	public PQOutput2[] getOutputs() {
		return outputExamples;
	}

	public PQInput2 getInput(int index) {
		return inputExamples[index];
	}

	public PQOutput2 getOutput(int index) {
		return outputExamples[index];
	}

	/**
	 * Load a dataset from the given stream.
	 * 
	 * @param fileName
	 *            the name of a file where to read from the dataset.
	 * 
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void load(String fileName) throws IOException, DatasetException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		load(reader);
		reader.close();
	}

	/**
	 * Load a dataset from the given stream.
	 * 
	 * @param is
	 *            an input stream from where the dataset is loaded.
	 * @throws DatasetException
	 * @throws IOException
	 */
	public void load(InputStream is) throws IOException, DatasetException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		load(reader);
	}

	/**
	 * Load a dataset from the given buffered reader.
	 * 
	 * @param reader
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		LinkedList<PQInput2> inputExamples = new LinkedList<PQInput2>();
		LinkedList<PQOutput2> outputExamples = new LinkedList<PQOutput2>();

		// Parse each example.
		int numTotal = 0;
		int numAdded = 0;
		String buff;
		while ((buff = skipBlanksAndComments(reader)) != null) {
			if (parseExample(inputExamples, outputExamples, buff))
				++numAdded;
			++numTotal;
		}

		LOG.info("Skipped " + (numTotal - numAdded) + " examples of "
				+ numTotal + " (" + (numTotal - numAdded) * 100d / numTotal
				+ "%)");

		this.inputExamples = inputExamples.toArray(new PQInput2[0]);
		this.outputExamples = outputExamples.toArray(new PQOutput2[0]);
	}

	/**
	 * Add the examples in the given dataset to this dataset.
	 * 
	 * @param other
	 */
	public void add(PQDataset2 other) throws DatasetException {
		if (!featureEncoding.equals(other.featureEncoding))
			throw new DatasetException("Different encodings");

		// Alloc room to store both datasets (this one and the given one).
		PQInput2[] newInputExamples = new PQInput2[inputExamples.length
				+ other.inputExamples.length];
		PQOutput2[] newOutputExamples = new PQOutput2[outputExamples.length
				+ other.outputExamples.length];

		// Copy (only reference) the examples in this dataset to the new arrays.
		int idx = 0;
		for (; idx < inputExamples.length; ++idx) {
			newInputExamples[idx] = inputExamples[idx];
			newOutputExamples[idx] = outputExamples[idx];
		}

		// Copy (only reference) the examples in the given dataset to the new
		// arrays.
		for (int idxO = 0; idxO < other.inputExamples.length; ++idxO, ++idx) {
			newInputExamples[idx] = other.inputExamples[idxO];
			newOutputExamples[idx] = other.outputExamples[idxO];
		}

		// Adjust the pointers of this dataset to the new arrays.
		this.inputExamples = newInputExamples;
		this.outputExamples = newOutputExamples;
	}

	/**
	 * Skip blank lines and lines starting by the comment character #.
	 * 
	 * @param reader
	 * @return
	 * @throws IOException
	 */
	protected String skipBlanksAndComments(BufferedReader reader)
			throws IOException {
		String buff;
		while ((buff = reader.readLine()) != null) {
			// Skip empty lines.
			if (buff.trim().length() == 0)
				continue;
			// Skip comment lines.
			if (buff.startsWith("#"))
				continue;
			break;
		}
		return buff;
	}

	/**
	 * Parse the given string and load an example.
	 * 
	 * @param buff
	 *            a string that contains an example.
	 * 
	 * @return <code>true</code> if the given string is a valid example.
	 * 
	 * @throws DatasetException
	 *             if there is some format problem with the given string.
	 */
	public boolean parseExample(Collection<PQInput2> exampleInputs,
			Collection<PQOutput2> exampleOutputs, String buff)
			throws DatasetException {
		// Split quotations.
		String quotationsInput[] = buff.split("§");
		if (quotationsInput.length == 0)
			return false;

		// First field: document ID.
		if (quotationsInput[0].trim().length() == 0)
			return false;
		String docId = quotationsInput[0];

		LinkedList<LinkedList<LinkedList<Integer>>> exampleInputAsList = new LinkedList<LinkedList<LinkedList<Integer>>>();
		LinkedList<Integer> exampleOutputAsList = new LinkedList<Integer>();
		LinkedList<Quotation> quotationsAsList = new LinkedList<Quotation>();

		// Walk into the document quotations.
		for (int idxQuote = 1; idxQuote < quotationsInput.length; idxQuote++) {
			// Split candidate coreferences of the given quotation.
			String coreferences[] = quotationsInput[idxQuote].split("\\t");

			if (coreferences.length == 0)
				return false;

			// First field: quotation start index, quotation end index,
			// right coreference index.
			String firstField[] = coreferences[0].split("[ ]");
			if (firstField.length != 3)
				return false;

			// Quotation start index, quotation end index.
			if ((firstField[0].trim().length() == 0)
					|| (firstField[1].trim().length() == 0))
				return false;

			Quotation quotation = new Quotation(coreferences.length - 1);
			quotation.setQuotationIndex(Integer.parseInt(firstField[0]),
					Integer.parseInt(firstField[1]));

			// Right coreference index.
			if (firstField[2].trim().length() == 0)
				return false;
			int rightCoref = Integer.parseInt(firstField[2]) + 1;

			// Walk into the coreference feature list.
			LinkedList<LinkedList<Integer>> corefFeatureList = new LinkedList<LinkedList<Integer>>();
			for (int idxCoref = 1; idxCoref < coreferences.length; ++idxCoref) {
				String coreference = coreferences[idxCoref];

				// Parse the given coreference features.
				String[] features = coreference.split("[ ]");
				if (firstField.length < 2)
					return false;

				// First field: coreference start index, coreference end index.
				if ((features[0].trim().length() == 0)
						|| (features[1].trim().length() == 0))
					return false;

				quotation.setCoreferenceIndex(idxCoref - 1,
						Integer.parseInt(features[0]),
						Integer.parseInt(features[1]));

				// Encode the features.
				LinkedList<Integer> featureList = new LinkedList<Integer>();
				for (int idxFtr = 2; idxFtr < features.length; ++idxFtr) {
					int code = featureEncoding.put(features[idxFtr]);
					if (code >= 0)
						featureList.add(code);
				}
				corefFeatureList.add(featureList);
			}

			// Example input.
			exampleInputAsList.add(corefFeatureList);

			// Example output.
			exampleOutputAsList.add(rightCoref);

			// Quotation index information.
			quotationsAsList.add(quotation);
		}

		// Store the loaded example.
		if (training) {
			/*
			 * Training examples must store internally their indexes in the
			 * array of training examples.
			 */
			exampleInputs.add(new PQInput2(docId, exampleInputAsList,
					quotationsAsList));
			exampleOutputs.add(new PQOutput2(exampleOutputAsList));
		} else {
			exampleInputs.add(new PQInput2(docId, exampleInputAsList,
					quotationsAsList));
			exampleOutputs.add(new PQOutput2(exampleOutputAsList));
		}
		return true;
	}

	/**
	 * Save this dataset in the given file.
	 * 
	 * @param fileName
	 *            name of the file to save this dataset.
	 * 
	 * @throws IOException
	 *             if some problem occurs when opening or writing to the file.
	 */
	public void save(String fileName) throws IOException {
		PrintStream ps = new PrintStream(fileName);
		save(ps);
		ps.close();
	}

	/**
	 * Save the dataset to the given stream.
	 * 
	 * @param os
	 *            an output stream to where the dataset is saved.
	 */
	public void save(PrintStream ps) {
		for (int idxExample = 0; idxExample < getNumberOfExamples(); ++idxExample) {
			PQInput2 input = inputExamples[idxExample];
			PQOutput2 output = outputExamples[idxExample];

			// The sentence identifier string.
			ps.print(input.getId());

			int numberOfQuotations = input.getNumberOfQuotations();
			for (int quotationIdx = 0; quotationIdx < numberOfQuotations; ++quotationIdx) {
				// Quotation separator.
				ps.print("§");

				int numberOfCoreferences = input
						.getNumberOfCoreferences(quotationIdx);
				for (int coreferenceIdx = 0; coreferenceIdx < numberOfCoreferences; ++coreferenceIdx) {
					// Coreference features.
					for (int ftr : input.getFeatureCodes(quotationIdx,
							coreferenceIdx))
						ps.print(featureEncoding.getValueByCode(ftr) + " ");

					// Coreference separator.
					ps.print("\t");
				}

				// Quotation author.
				ps.println(featureEncoding.getValueByCode(output
						.getAuthor(quotationIdx)));
			}

			// Next line for the next example.
			ps.println();
		}
	}

	/**
	 * Return the number of symbols in the dataset feature-value encoding
	 * object. In general, this corresponds to the total number of different
	 * symbols in the dataset, but can be a different number if the encoding was
	 * used by other code despite this dataset.
	 * 
	 * @return
	 */
	public int getNumberOfSymbols() {
		return featureEncoding.size();
	}

	/**
	 * Normalize all the input structures within this dataset to have the same
	 * given norm.
	 * 
	 * @param norm
	 */
	public void normalizeInputStructures(double norm) {
		for (PQInput2 in : inputExamples)
			in.normalize(norm);
	}

	/**
	 * Sort feature values of each token to speedup kernel functions
	 * computations.
	 */
	public void sortFeatureValues() {
		for (PQInput2 seq : inputExamples)
			seq.sortFeatures();
	}

}
