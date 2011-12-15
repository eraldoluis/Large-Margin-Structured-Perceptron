package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a textual dataset whose examples are sequence of tokens.
 * 
 * The feature values are stored as integers and are also called feature codes.
 * Different schema can be used to encode textual feature values to integer
 * codes by specializing the class <code>FeatureEncoding</code>. The default
 * feature encoding scheme is <code>StringMapEncoding</code>. It stores an array
 * of strings and use the index in this array as feature code. A
 * <code>HashMap</code> stores the inverted index, i.e., efficiently provides
 * the code of a feature value given its textual representation.
 * 
 */
public class SequenceDataset implements Dataset {

	/**
	 * Loging object.
	 */
	private static final Log LOG = LogFactory.getLog(SequenceDataset.class);

	/**
	 * Special code used to indicate non-annotated tokens.
	 */
	public static final int NON_ANNOTATED_STATE_CODE = -10;

	/**
	 * Special code used to indicate unknown token labels.
	 */
	public static final int UNKNOWN_STATE_CODE = -20;

	/**
	 * Special code used to indicate unknown features.
	 */
	public static final int UNKNOWN_FEATURE_CODE = -30;

	/**
	 * Indicate whether this is a training dataset or not.
	 */
	protected boolean training;

	/**
	 * Map string feature values to integer values (codes).
	 */
	protected FeatureEncoding<String> featureEncoding;

	/**
	 * Map string state values to integer values (codes).
	 */
	protected FeatureEncoding<String> stateEncoding;

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
	 * If <code>true</code>, when loading partially-labeled datasets, skip
	 * examples that do not include any label.
	 */
	protected boolean skipCompletelyNonAnnotatedExamples;

	/**
	 * Store the maximum number of emission features seen in a unique token in
	 * the dataset.
	 */
	protected int maxNumberOfEmissionFeatures;

	/**
	 * Default constructor.
	 */
	public SequenceDataset() {
		this(new StringMapEncoding(), new StringMapEncoding());
	}

	/**
	 * Create a dataset using the given feature-value and state-label encodings.
	 * One can use this constructor to create a dataset compatible with a
	 * previous loaded model, for instance.
	 * 
	 * @param featureEncoding
	 * @param stateEncoding
	 */
	public SequenceDataset(FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		this.featureEncoding = featureEncoding;
		this.stateEncoding = stateEncoding;
		this.skipCompletelyNonAnnotatedExamples = false;
		this.training = false;
	}

	public SequenceDataset(FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding, String nonAnnotatedStateLabel) {
		this.featureEncoding = featureEncoding;
		this.stateEncoding = stateEncoding;
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.skipCompletelyNonAnnotatedExamples = false;
		this.training = false;
	}

	public SequenceDataset(FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding,
			String nonAnnotatedStateLabel, boolean training) {
		this.featureEncoding = featureEncoding;
		this.stateEncoding = stateEncoding;
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.skipCompletelyNonAnnotatedExamples = false;
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
	public SequenceDataset(String fileName) throws IOException,
			DatasetException {
		this(new StringMapEncoding(), new StringMapEncoding());
		load(fileName);
	}

	public SequenceDataset(String fileName, String nonAnnotatedStateLabel)
			throws IOException, DatasetException {
		this(new StringMapEncoding(), new StringMapEncoding());
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		load(fileName);
	}

	/**
	 * Load the dataset from a <code>InputStream</code>.
	 * 
	 * @param is
	 * @throws IOException
	 * @throws DatasetException
	 */
	public SequenceDataset(InputStream is) throws IOException, DatasetException {
		this(new StringMapEncoding(), new StringMapEncoding());
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
	public SequenceDataset(String fileName,
			FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) throws IOException,
			DatasetException {
		this(featureEncoding, stateEncoding);
		load(fileName);
	}

	public SequenceDataset(String fileName,
			FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding, String nonAnnotatedStateLabel)
			throws IOException, DatasetException {
		this(featureEncoding, stateEncoding);
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		load(fileName);
	}

	public SequenceDataset(String fileName,
			FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding,
			String nonAnnotatedStateLabel,
			boolean skipCompletelyNonAnnotatedExamples) throws IOException,
			DatasetException {
		this(featureEncoding, stateEncoding);
		this.nonAnnotatedStateLabel = nonAnnotatedStateLabel;
		this.skipCompletelyNonAnnotatedExamples = skipCompletelyNonAnnotatedExamples;
		load(fileName);
	}

	@Override
	public boolean isTraining() {
		return training;
	}

	@Override
	public int getNumberOfExamples() {
		return inputSequences.length;
	}

	@Override
	public SequenceInput[] getInputs() {
		return inputSequences;
	}

	@Override
	public SequenceOutput[] getOutputs() {
		return outputSequences;
	}

	@Override
	public SequenceInput getInput(int index) {
		return inputSequences[index];
	}

	@Override
	public SequenceOutput getOutput(int index) {
		return outputSequences[index];
	}

	/**
	 * Return the encoding scheme for feature values.
	 * 
	 * @return
	 */
	public FeatureEncoding<String> getFeatureEncoding() {
		return featureEncoding;
	}

	/**
	 * Return the encoding scheme for state labels.
	 * 
	 * @return
	 */
	public FeatureEncoding<String> getStateEncoding() {
		return stateEncoding;
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
		LinkedList<ArraySequenceInput> inputSequences = new LinkedList<ArraySequenceInput>();
		LinkedList<ArraySequenceOutput> outputSequences = new LinkedList<ArraySequenceOutput>();

		// Parse each example.
		int numTotal = 0;
		int numAdded = 0;
		String buff;
		while ((buff = skipBlanksAndComments(reader)) != null) {
			if (parseExample(inputSequences, outputSequences, buff))
				++numAdded;
			++numTotal;
		}

		LOG.info("Skipped " + (numTotal - numAdded) + " examples of "
				+ numTotal + " (" + (numTotal - numAdded) * 100d / numTotal
				+ "%)");

		this.inputSequences = inputSequences.toArray(new SequenceInput[0]);
		this.outputSequences = outputSequences.toArray(new SequenceOutput[0]);
	}

	/**
	 * Add the examples in the given dataset to this dataset.
	 * 
	 * @param other
	 */
	public void add(SequenceDataset other) throws DatasetException {
		if (!featureEncoding.equals(other.featureEncoding)
				|| !stateEncoding.equals(other.stateEncoding))
			throw new DatasetException("Different encodings");

		// Alloc room to store both datasets (this one and the given one).
		SequenceInput[] newInputSequences = new SequenceInput[inputSequences.length
				+ other.inputSequences.length];
		SequenceOutput[] newOutputSequences = new SequenceOutput[outputSequences.length
				+ other.outputSequences.length];

		// Copy (only reference) the examples in this dataset to the new arrays.
		int idx = 0;
		for (; idx < inputSequences.length; ++idx) {
			newInputSequences[idx] = inputSequences[idx];
			newOutputSequences[idx] = outputSequences[idx];
		}

		// Copy (only reference) the examples in the given dataset to the new
		// arrays.
		for (int idxO = 0; idxO < other.inputSequences.length; ++idxO, ++idx) {
			newInputSequences[idx] = other.inputSequences[idxO];
			newOutputSequences[idx] = other.outputSequences[idxO];
		}

		// Adjust the pointers of this dataset to the new arrays.
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
	public boolean parseExample(Collection<ArraySequenceInput> sequenceInputs,
			Collection<ArraySequenceOutput> sequenceOutputs, String buff)
			throws DatasetException {
		// Split tokens.
		String tokens[] = buff.split("\\t");

		if (tokens.length == 0)
			return false;

		// The first field is the sentence id.
		String id = tokens[0];

		if (id.trim().length() == 0)
			return false;

		LinkedList<LinkedList<Integer>> sequenceInputAsList = new LinkedList<LinkedList<Integer>>();
		LinkedList<Integer> sequenceOutputAsList = new LinkedList<Integer>();

		boolean someAnnotatedToken = false;

		for (int idxTkn = 1; idxTkn < tokens.length; ++idxTkn) {
			String token = tokens[idxTkn];

			// Parse the token features.
			int numEmissionFeatures = 0;
			String[] features = token.split("[ ]");
			LinkedList<Integer> featureList = new LinkedList<Integer>();
			for (int idxFtr = 0; idxFtr < features.length - 1; ++idxFtr) {
				++numEmissionFeatures;
				int code = featureEncoding.put(new String(features[idxFtr]));
				if (code >= 0)
					featureList.add(code);
			}

			if (numEmissionFeatures > maxNumberOfEmissionFeatures)
				maxNumberOfEmissionFeatures = numEmissionFeatures;

			// The last feature is the token label.
			String label = features[features.length - 1];
			if (label.equals(nonAnnotatedStateLabel))
				// The label indicates a non-annotated token and then we use the
				// non-annotated state code, instead of encoding this special
				// label. Note that the above test always returns false if the
				// special label is null (totally annotated dataset).
				sequenceOutputAsList.add(NON_ANNOTATED_STATE_CODE);
			else {
				int code = stateEncoding.put(new String(label));
				if (code < 0)
					LOG.warn("Unknown label (" + label + ") in token "
							+ (idxTkn - 1) + " of example " + id
							+ " is unknown");
				sequenceOutputAsList.add(code);
				someAnnotatedToken = true;
			}

			sequenceInputAsList.add(featureList);
		}

		// Store the loaded example.
		if (!skipCompletelyNonAnnotatedExamples || someAnnotatedToken) {
			if (training) {
				/*
				 * Training examples must store internally their indexes in the
				 * array of training examples.
				 */
				sequenceInputs.add(new ArraySequenceInput(new String(id), sequenceInputs
						.size(), sequenceInputAsList));
				sequenceOutputs.add(new ArraySequenceOutput(
						sequenceOutputAsList, sequenceOutputAsList.size()));
			} else {
				sequenceInputs.add(new ArraySequenceInput(new String(id),
						sequenceInputAsList));
				sequenceOutputs.add(new ArraySequenceOutput(
						sequenceOutputAsList, sequenceOutputAsList.size()));
			}
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
			SequenceInput input = inputSequences[idxSequence];
			SequenceOutput output = outputSequences[idxSequence];

			// The sentence identifier string.
			ps.print(input.getId());

			for (int token = 0; token < input.size(); ++token) {
				// Tokens as separated.
				ps.print("\t");

				// Token features.
				for (int ftr : input.getFeatureCodes(token))
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

	/**
	 * Normalize all the input structures within this dataset to have the same
	 * given norm.
	 * 
	 * @param norm
	 */
	public void normalizeInputStructures(double norm) {
		for (SequenceInput in : inputSequences)
			in.normalize(norm);
	}

	/**
	 * Return the maximum number of emission features seen in a unique token.
	 * 
	 * @return
	 */
	public int getMaxNumberOfEmissionFeatures() {
		return maxNumberOfEmissionFeatures;
	}

	/**
	 * Sort feature values of each token to speedup kernel functions
	 * computations.
	 */
	public void sortFeatureValues() {
		for (SequenceInput seq : inputSequences)
			seq.sortFeatures();
	}

	@Override
	public void save(BufferedWriter writer) throws IOException,
			DatasetException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void save(OutputStream os) throws IOException, DatasetException {
		// TODO Auto-generated method stub
		
	}

}
