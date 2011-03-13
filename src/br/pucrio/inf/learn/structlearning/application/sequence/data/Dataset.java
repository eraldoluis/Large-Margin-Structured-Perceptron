package br.pucrio.inf.learn.structlearning.application.sequence.data;

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

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;

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
public class Dataset {

	private static final Log LOG = LogFactory.getLog(Dataset.class);

	/**
	 * Map string feature values to integer values (codes).
	 */
	protected StringEncoding featureEncoding;

	/**
	 * Map string state values to integer values (codes).
	 */
	protected StringEncoding stateEncoding;

	/**
	 * IDs of the examples.
	 */
	protected String[] exampleIDs;

	/**
	 * Vector of the input-part of the examples.
	 */
	protected SequenceInput[] inputSequences;

	/**
	 * Vector of the output-part of the examples (correct predictions).
	 */
	protected SequenceOutput[] outputSequences;

	/**
	 * Special label that indicates non-annotated tokens.
	 */
	protected String nonAnnotatedStateLabel;

	/**
	 * Invalid state code to be used for the non-annotated tokens.
	 */
	protected int nonAnnotatedStateCode;

	/**
	 * If <code>true</code>, when loading partially-labeled datasets, skip
	 * examples that do not include any label.
	 */
	protected boolean skipCompletelyNonAnnotatedExamples;

	/**
	 * Default constructor.
	 */
	public Dataset() {
		this(new StringEncoding(), new StringEncoding());
	}

	/**
	 * Create a dataset using the given feature-value and state-label encodings.
	 * One can use this constructor to create a dataset compatible with a
	 * previous loaded model, for instance.
	 * 
	 * @param featureEncoding
	 * @param stateEncoding
	 */
	public Dataset(StringEncoding featureEncoding, StringEncoding stateEncoding) {
		this.featureEncoding = featureEncoding;
		this.stateEncoding = stateEncoding;
		this.skipCompletelyNonAnnotatedExamples = false;
	}

	public Dataset(StringEncoding featureEncoding,
			StringEncoding stateEncoding, String nonAnnotatedStateLabel,
			int nonAnnotatedStateCode) {
		this.featureEncoding = featureEncoding;
		this.stateEncoding = stateEncoding;
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.nonAnnotatedStateCode = nonAnnotatedStateCode;
		this.skipCompletelyNonAnnotatedExamples = false;
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
	public Dataset(String fileName) throws IOException, DatasetException {
		this(new StringEncoding(), new StringEncoding());
		load(fileName);
	}

	public Dataset(String fileName, String nonAnnotatedStateLabel,
			int nonAnnotatedStateCode) throws IOException, DatasetException {
		this(new StringEncoding(), new StringEncoding());
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.nonAnnotatedStateCode = nonAnnotatedStateCode;
		load(fileName);
	}

	/**
	 * Load the dataset from a <code>InputStream</code>.
	 * 
	 * @param is
	 * @throws IOException
	 * @throws DatasetException
	 */
	public Dataset(InputStream is) throws IOException, DatasetException {
		this(new StringEncoding(), new StringEncoding());
		load(is);
	}

	/**
	 * Load the dataset from the given file and use the given feature-value and
	 * state-label encodings. One can use this constructor to load a dataset
	 * compatible with a previous loaded model, for instance.
	 * 
	 * @param fileName
	 *            name and path of a file.
	 * @param featureEncoding
	 *            use a determined feature values encoding.
	 * @param stateEncoding
	 *            use a determined state labels encoding.
	 * 
	 * @throws IOException
	 *             if occurs some problem reading the file.
	 * @throws DatasetException
	 *             if the file contains invalid data.
	 */
	public Dataset(String fileName, StringEncoding featureEncoding,
			StringEncoding stateEncoding) throws IOException, DatasetException {
		this(featureEncoding, stateEncoding);
		load(fileName);
	}

	public Dataset(String fileName, StringEncoding featureEncoding,
			StringEncoding stateEncoding, String nonAnnotatedStateLabel,
			int nonAnnotateStateCode) throws IOException, DatasetException {
		this(featureEncoding, stateEncoding);
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.nonAnnotatedStateCode = nonAnnotateStateCode;
		load(fileName);
	}

	public Dataset(String fileName, StringEncoding featureEncoding,
			StringEncoding stateEncoding, String nonAnnotatedStateLabel,
			int nonAnnotateStateCode, boolean skipCompletelyNonAnnotatedExamples)
			throws IOException, DatasetException {
		this(featureEncoding, stateEncoding);
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.nonAnnotatedStateCode = nonAnnotateStateCode;
		this.skipCompletelyNonAnnotatedExamples = skipCompletelyNonAnnotatedExamples;
		load(fileName);
	}

	public SequenceInput getSequenceInput(int index) {
		return inputSequences[index];
	}

	/**
	 * Return the number of examples within this dataset.
	 * 
	 * @return the number of examples within this dataset
	 */
	public int getNumberOfExamples() {
		return inputSequences.length;
	}

	public StringEncoding getFeatureEncoding() {
		return featureEncoding;
	}

	public StringEncoding getStateEncoding() {
		return stateEncoding;
	}

	public SequenceInput[] getInputs() {
		return inputSequences;
	}

	public SequenceOutput[] getOutputs() {
		return outputSequences;
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
		LinkedList<String> ids = new LinkedList<String>();
		LinkedList<SequenceInput> inputSequences = new LinkedList<SequenceInput>();
		LinkedList<SequenceOutput> outputSequences = new LinkedList<SequenceOutput>();

		// Parse each example.
		int numTotal = 0;
		int numAdded = 0;
		String buff;
		while ((buff = skipBlanksAndComments(reader)) != null) {
			if (parseExample(ids, inputSequences, outputSequences, buff))
				++numAdded;
			++numTotal;
		}

		LOG.info("Skipped " + (numTotal - numAdded) + " examples of "
				+ numTotal + " (" + (numTotal - numAdded) * 100d / numTotal
				+ "%)");

		this.exampleIDs = ids.toArray(new String[0]);
		this.inputSequences = inputSequences.toArray(new SequenceInput[0]);
		this.outputSequences = outputSequences.toArray(new SequenceOutput[0]);
	}

	/**
	 * Add the examples in the given dataset to this dataset.
	 * 
	 * @param other
	 */
	public void add(Dataset other) throws DatasetException {
		if (!featureEncoding.equals(other.featureEncoding)
				|| !stateEncoding.equals(other.stateEncoding))
			throw new DatasetException("Different encodings");

		// Alloc room to store both datasets (this one and the given one).
		String[] newExampleIDs = new String[exampleIDs.length
				+ other.exampleIDs.length];
		SequenceInput[] newInputSequences = new SequenceInput[inputSequences.length
				+ other.inputSequences.length];
		SequenceOutput[] newOutputSequences = new SequenceOutput[outputSequences.length
				+ other.outputSequences.length];

		// Copy (only reference) the examples in this dataset to the new arrays.
		int idx = 0;
		for (; idx < inputSequences.length; ++idx) {
			newExampleIDs[idx] = exampleIDs[idx];
			newInputSequences[idx] = inputSequences[idx];
			newOutputSequences[idx] = outputSequences[idx];
		}

		// Copy (only reference) the examples in the given dataset to the new
		// arrays.
		for (int idxO = 0; idxO < other.exampleIDs.length; ++idxO, ++idx) {
			newExampleIDs[idx] = other.exampleIDs[idxO];
			newInputSequences[idx] = other.inputSequences[idxO];
			newOutputSequences[idx] = other.outputSequences[idxO];
		}

		// Adjust the pointers of this dataset to the new arrays.
		this.exampleIDs = newExampleIDs;
		this.inputSequences = newInputSequences;
		this.outputSequences = newOutputSequences;
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
	public boolean parseExample(Collection<String> ids,
			Collection<SequenceInput> sequenceInputs,
			Collection<SequenceOutput> sequenceOutputs, String buff)
			throws DatasetException {
		// Split tokens.
		String tokens[] = buff.split("\\t");

		if (tokens.length == 0)
			return false;

		// The first field is the sentence id.
		String id = tokens[0];

		if (id.trim().length() == 0)
			return false;

		ids.add(id);

		LinkedList<LinkedList<Integer>> sequenceInputAsList = new LinkedList<LinkedList<Integer>>();
		LinkedList<Integer> sequenceOutputAsList = new LinkedList<Integer>();

		boolean someAnnotatedToken = false;

		for (int idxTkn = 1; idxTkn < tokens.length; ++idxTkn) {
			String token = tokens[idxTkn];

			// Parse the token features.
			String[] features = token.split("[ ]");
			LinkedList<Integer> featureList = new LinkedList<Integer>();
			for (int idxFtr = 0; idxFtr < features.length - 1; ++idxFtr) {
				int code = featureEncoding.put(features[idxFtr]);
				if (code >= 0)
					featureList.add(code);
			}

			// The last feature is the token label.
			String label = features[features.length - 1];
			if (label.equals(nonAnnotatedStateLabel))
				// The label indicates a non-annotated token and then we use the
				// non-annotated state code, instead of encoding this special
				// label. Note that the above test always returns false if the
				// special label is null (totally annotated dataset).
				sequenceOutputAsList.add(nonAnnotatedStateCode);
			else {
				sequenceOutputAsList.add(stateEncoding.put(label));
				someAnnotatedToken = true;
			}

			sequenceInputAsList.add(featureList);
		}

		// Store the loaded example.
		if (!skipCompletelyNonAnnotatedExamples || someAnnotatedToken) {
			sequenceInputs.add(new SequenceInput(sequenceInputAsList));
			sequenceOutputs.add(new SequenceOutput(sequenceOutputAsList,
					sequenceOutputAsList.size()));
			return true;
		}

		return false;
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
		for (int idxSequence = 0; idxSequence < getNumberOfExamples(); ++idxSequence) {
			String id = exampleIDs[idxSequence];
			SequenceInput input = inputSequences[idxSequence];
			SequenceOutput output = outputSequences[idxSequence];

			// The sentence identifier string.
			ps.print(id);

			for (int token = 0; token < input.size(); ++token) {
				// Tokens as separated.
				ps.print("\t");

				// Token features.
				for (int ftr : input.getFeatures(token))
					ps.print(featureEncoding.getValueByCode(ftr) + " ");

				// Label of this token.
				ps.println(featureEncoding.getValueByCode(output
						.getLabel(token)));
			}

			// Next line for the next sequence.
			ps.println();
		}
	}

	/**
	 * Return the number of tokens within the given example index.
	 * 
	 * @param idxExample
	 *            index of an example.
	 * 
	 * @return the number of tokens within the given example index.
	 */
	public int getNumberOfTokens(int idxExample) {
		return inputSequences[idxExample].size();
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
	 * Return the number of different state labels within the dataset
	 * state-label encoding. In general, this corresponds to the total number of
	 * different state labels in the dataset, but can be a different number if
	 * the encoding was used by other code despite this dataset.
	 * 
	 * @return
	 */
	public int getNumberOfStates() {
		return stateEncoding.size();
	}

	/**
	 * If set to <code>true</code>, skip examples with no labeled token.
	 * 
	 * @param skipCompletelyNonAnnotatedExamples
	 */
	public void setSkipCompletelyNonAnnotatedExamples(
			boolean skipCompletelyNonAnnotatedExamples) {
		this.skipCompletelyNonAnnotatedExamples = skipCompletelyNonAnnotatedExamples;
	}

}
