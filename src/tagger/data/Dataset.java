package tagger.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

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
public class Dataset implements Iterable<DatasetExample> {

	/**
	 * Control the mapping from string feature values to integer values and
	 * vice-versa.
	 */
	protected FeatureValueEncoding featureValueEncoding;

	/**
	 * The feature labels.
	 */
	protected Vector<String> featureLabels;

	/**
	 * IDs of the examples.
	 */
	protected Vector<String> exampleIDs;

	/**
	 * The encoded data, i.e., integer values of the features.
	 */
	protected Vector<Vector<Vector<Integer>>> examples;

	// TODO this prevents the access of different examples at the same time.
	protected Example tempExample = new Example(0);

	/**
	 * Default constructor.
	 */
	public Dataset() {
		featureValueEncoding = new FeatureValueEncoding();
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>();
		exampleIDs = new Vector<String>();
	}

	/**
	 * Create a dataset using the given feature-value encoding.
	 * 
	 * @param featureValueEncoding
	 */
	public Dataset(FeatureValueEncoding featureValueEncoding) {
		this.featureValueEncoding = featureValueEncoding;
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>();
		exampleIDs = new Vector<String>();
	}

	/**
	 * Create a dataset from file.
	 * 
	 * @param fileName
	 *            the name of a file to load the dataset.
	 * 
	 * @throws DatasetException
	 * @throws IOException
	 */
	public Dataset(String fileName) throws IOException, DatasetException {
		featureValueEncoding = new FeatureValueEncoding();
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>();
		exampleIDs = new Vector<String>();
		load(fileName);
	}

	public Dataset(InputStream is) throws IOException, DatasetException {
		featureValueEncoding = new FeatureValueEncoding();
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>();
		exampleIDs = new Vector<String>();
		load(is);
	}

	/**
	 * Load the dataset from the given file and use the given feature-value
	 * encoding.
	 * 
	 * @param fileName
	 *            name and path of a file.
	 * @param featureValueEncoding
	 *            a feature-value encoding.
	 * 
	 * @throws IOException
	 *             if occurs some problem reading the file.
	 * @throws DatasetException
	 *             if the file contains invalid data.
	 */
	public Dataset(String fileName, FeatureValueEncoding featureValueEncoding)
			throws IOException, DatasetException {
		this.featureValueEncoding = featureValueEncoding;
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>();
		exampleIDs = new Vector<String>();
		load(fileName);
	}

	/**
	 * Return a stub for the example at the given index.
	 * 
	 * @param index
	 *            the index of the wanted example.
	 * 
	 * @return the example stub.
	 */
	public DatasetExample getExample(int index) {
		tempExample.idxExample = index;
		return tempExample;
	}

	/**
	 * Randomly split the dataset into two parts according to the given
	 * proportion.
	 * 
	 * @param propor
	 *            a value between 1 and 99 that determines the proportion of
	 *            examples for each split.
	 * 
	 * @return a 2-element array with the two splits.
	 */
	public Dataset[] split(int propor) {
		return null;
	}

	/**
	 * Add the examples in the given dataset to this dataset.
	 * 
	 * @param dataset
	 *            a <code>Dataset</code> to be joined to this dataset.
	 * 
	 * @throws DatasetException
	 *             if the given dataset has a different shape or a different
	 *             feature-value encoding.
	 */
	public void add(Dataset dataset) throws DatasetException {
		if (dataset.getNumberOfFeatures() != getNumberOfFeatures())
			throw new DatasetException(
					"Both datasets need to have the same number of features.");

		if (dataset.featureValueEncoding != featureValueEncoding)
			throw new DatasetException(
					"Both datasets need to have the same feature-value encoding.");

		examples.addAll(dataset.examples);
		exampleIDs.addAll(dataset.exampleIDs);
	}

	/**
	 * Return the number of examples within this dataset.
	 * 
	 * @return the number of examples within this dataset
	 */
	public int getNumberOfExamples() {
		return examples.size();
	}

	/**
	 * Return the number of features in this dataset.
	 * 
	 * @return the number of features in this dataset.
	 */
	public int getNumberOfFeatures() {
		return featureLabels.size();
	}

	/**
	 * Create a new feature in this dataset and adapt its shape to the new
	 * feature (filling the new feature value with <code>null</code> in all
	 * examples).
	 * 
	 * @param label
	 *            the label of the new feature.
	 * 
	 * @return the index of the created feature.
	 * 
	 * @throws DatasetException
	 *             if the given label already exists.
	 */
	public int createNewFeature(String label) throws DatasetException {
		// Verify if the feature label already is used.
		if (featureLabels.contains(label))
			throw new DatasetException("Feature " + label
					+ " already exists in this dataset.");

		// Add the new feature label to the list.
		featureLabels.add(label);

		// Create a new element in all token vectors.
		adjustDataShape(1);

		// Return the new feature index.
		return featureLabels.size() - 1;
	}

	/**
	 * Remove the feature with the given label.
	 * 
	 * @param label
	 * @throws DatasetException
	 *             if the given there is no feature with the given label.
	 */
	public void removeFeature(String label) throws DatasetException {
		// Feature index.
		int feature = getFeatureIndex(label);
		if (feature < 0)
			throw new DatasetException("Inexistent feature label: " + label);

		// Remove the feature from the label vector.
		featureLabels.remove(feature);

		// Remove the feature from every token.
		for (Vector<Vector<Integer>> example : examples)
			for (Vector<Integer> token : example)
				token.remove(feature);
	}

	/**
	 * Create new features with the given labels and adjust the dataset shape to
	 * accommodate them (filling with <code>null</code> the new-features values
	 * in all the examples).
	 * 
	 * @param labels
	 *            the labels of the new features.
	 * 
	 * @return the index of each new feature.
	 * 
	 * @throws DatasetException
	 *             if some of the given labels already exists.
	 */
	public int[] createNewFeatures(String[] labels) throws DatasetException {
		// Verify if all labels are ok (new ones).
		for (String label : labels) {
			if (featureLabels.contains(label))
				throw new DatasetException("Feature " + label
						+ " already exists in this dataset.");
		}

		// Add new elements in all token vectors.
		adjustDataShape(labels.length);

		// Add the feature labels.
		int idx = 0;
		int[] featureIndexes = new int[labels.length];
		for (String label : labels) {
			featureIndexes[idx++] = featureLabels.size();
			featureLabels.add(label);
		}

		return featureIndexes;
	}

	/**
	 * Create new elements in all token vectors to accomodate new features.
	 * 
	 * @param numberOfExtraFeatures
	 *            the number of new features to be created.
	 */
	private void adjustDataShape(int numberOfExtraFeatures) {
		for (Vector<Vector<Integer>> example : examples)
			for (Vector<Integer> token : example)
				token.setSize(token.size() + numberOfExtraFeatures);
	}

	/**
	 * Create a new example with the given feature matrix.
	 * 
	 * The matrix values must be feature values within the
	 * <code>FeatureValueEncoding</code> of this dataset.
	 * 
	 * @param id
	 *            the identification string of the new example.
	 * @param exampleFeatures
	 *            the feature values of the new example.
	 * 
	 * @return a <code>DatasetExample</code> representing the new created
	 *         example.
	 * 
	 * @throws DatasetException
	 *             if the shape (number of features) of the given matrix does
	 *             not reflects the shape of this dataset.
	 */
	public DatasetExample addExample(String id,
			Collection<? extends Collection<Integer>> exampleFeatures)
			throws DatasetException {

		Vector<Vector<Integer>> example = new Vector<Vector<Integer>>(
				exampleFeatures.size());

		for (Collection<Integer> token : exampleFeatures) {
			if (token.size() != getNumberOfFeatures())
				throw new DatasetException(
						"The given example has a different number of features than this dataset.");

			// Create a new vector representing a token and add it to the
			// example.
			Vector<Integer> tokenV = new Vector<Integer>(getNumberOfFeatures());
			example.add(tokenV);

			// Fill the token feature values.
			for (int ftr : token)
				tokenV.add(ftr);
		}

		// Add the example to the dataset.
		examples.add(example);
		exampleIDs.add(id);

		return new Example(getNumberOfExamples() - 1);
	}

	/**
	 * Add a new example, represented as a String vector, in the dataset.
	 * 
	 * The example is given by its features labels. These labels are encoded as
	 * integer values using the <code>FeatureValueEncoding</code> of this
	 * dataset.
	 * 
	 * @param id
	 *            the identification string of the new example.
	 * @param exampleFeatureLabels
	 *            the example as a matrix of strings.
	 * 
	 * @return a <code>DatasetExample</code> representing the new example.
	 * 
	 * @throws DatasetException
	 *             if the shape (number of features) of the given example is
	 *             different from the shape of this dataset.
	 */
	public DatasetExample addExampleAsString(String id,
			Collection<? extends Collection<String>> exampleFeatureLabels)
			throws DatasetException {

		Vector<Vector<Integer>> example = new Vector<Vector<Integer>>(
				exampleFeatureLabels.size());

		for (Collection<String> token : exampleFeatureLabels) {
			if (token.size() != getNumberOfFeatures())
				throw new DatasetException(
						"The given example has a different number of features than this dataset.");

			// Create a new vector representing a token and add it to the
			// example.
			Vector<Integer> tokenV = new Vector<Integer>(getNumberOfFeatures());
			example.add(tokenV);

			// Fill the token feature values.
			for (String ftrLabel : token) {
				int ftr = featureValueEncoding.putString(ftrLabel);
				tokenV.add(ftr);
			}
		}

		// Add the example to the dataset.
		examples.add(example);
		exampleIDs.add(id);

		return new Example(getNumberOfExamples() - 1);
	}

	/**
	 * Return the index of the feature with the given label.
	 * 
	 * @param label
	 *            the label of a feature.
	 * 
	 * @return the index of a feature or <code>-1</code> if the label does not
	 *         exist.
	 */
	public int getFeatureIndex(String label) {
		return featureLabels.indexOf(label);
	}

	/**
	 * Return the label of the feature with the given index.
	 * 
	 * @param index
	 *            the index of a feature.
	 * 
	 * @return the label of a feature.
	 */
	public String getFeatureLabel(int index) {
		return featureLabels.get(index);
	}

	public FeatureValueEncoding getFeatureValueEncoding() {
		return featureValueEncoding;
	}

	@Override
	public Iterator<DatasetExample> iterator() {
		return new DatasetIterator();
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
		// Search for the feature labels line.
		String buff = skipBlanksAndComments(reader, "# feature labels");
		if (buff == null)
			return;

		// Parse the feature labels.
		parseFeatureLabels(buff);

		// Parse each example.
		while ((buff = skipBlanksAndComments(reader)) != null)
			parseExample(buff);
	}

	public void loadWithoutHeader(String fileName) throws IOException,
			DatasetException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		loadWithoutHeader(reader);
		reader.close();
	}

	private void loadWithoutHeader(BufferedReader reader)
			throws DatasetException, IOException {
		String buff;
		// Parse each example.
		while ((buff = skipBlanksAndComments(reader)) != null)
			parseExample(buff);
	}

	protected String skipBlanksAndComments(BufferedReader reader)
			throws IOException {
		String buff;
		while ((buff = reader.readLine()) != null
				&& (buff.trim().length() == 0 || buff.startsWith("#")))
			;
		return buff;
	}

	protected String skipBlanksAndComments(BufferedReader reader,
			String stopOnThis) throws IOException {
		String buff;
		while ((buff = reader.readLine()) != null) {
			if (buff.equals(stopOnThis)) {
				buff = reader.readLine();
				break;
			} else if (buff.trim().length() > 0 && !buff.startsWith("#"))
				break;
		}
		return buff;
	}

	/**
	 * Parse the feature labels line.
	 * 
	 * @param buff
	 *            a string containing the feature labels.
	 * 
	 * @throws DatasetException
	 */
	private void parseFeatureLabels(String buff) {
		String[] labels = buff.split("[\\s]");
		try {
			createNewFeatures(labels);
		} catch (DatasetException e) {
			// This never must happen.
			assert false;
		}
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
	public boolean parseExample(String buff) throws DatasetException {
		// Split tokens.
		String tokens[] = buff.split("[\t]");

		if (tokens.length == 0)
			return false;

		// The first field is the sentence id.
		String id = tokens[0];

		if (id.trim().length() == 0)
			return false;

		Vector<Vector<Integer>> exampleData = new Vector<Vector<Integer>>(
				tokens.length - 1);

		for (int idxTkn = 1; idxTkn < tokens.length; ++idxTkn) {
			String token = tokens[idxTkn];

			// Split the token into its features.
			String[] features = token.split("[ ]");
			if (features.length != getNumberOfFeatures())
				throw new DatasetException(
						"Incorrect number of features on the following example:\n"
								+ buff);

			// Encode the feature values.
			Vector<Integer> featuresAsVector = new Vector<Integer>(
					features.length);
			for (String ftr : features)
				featuresAsVector.add(featureValueEncoding.putString(ftr));

			exampleData.add(featuresAsVector);
		}

		// Store the loaded example.
		exampleIDs.add(id);
		examples.add(exampleData);

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
		// Header.
		ps.println("# feature labels");
		for (String ftrLabel : featureLabels)
			ps.print(ftrLabel + " ");
		ps.println("\n");

		ps.println("# examples");
		for (int idxSent = 0; idxSent < getNumberOfExamples(); ++idxSent) {
			DatasetExample example = getExample(idxSent);

			// The sentence identifier string.
			ps.print(example.getID());

			for (int idxTkn = 0; idxTkn < example.size(); ++idxTkn) {
				// Tokens as separated.
				ps.print("\t");

				// Token features.
				for (int ftr = 0; ftr < getNumberOfFeatures(); ++ftr)
					ps.print(example.getFeatureValueAsString(idxTkn, ftr) + " ");
			}

			// Next line for the next sentence.
			ps.println();
		}

		ps.flush();
	}

	/**
	 * Save this dataset to the given fileName using the CoNLL format.
	 * 
	 * @param fileName
	 * @throws IOException
	 */
	public void saveAsCoNLL(String fileName) throws IOException {
		PrintStream ps = new PrintStream(fileName);
		saveAsCoNLL(ps);
		ps.close();
	}

	/**
	 * Save this dataset to the given stream using the CoNLL format.
	 * 
	 * @param ps
	 */
	public void saveAsCoNLL(PrintStream ps) {
		for (int idxSent = 0; idxSent < getNumberOfExamples(); ++idxSent) {
			DatasetExample example = getExample(idxSent);

			for (int idxTkn = 0; idxTkn < example.size(); ++idxTkn) {
				// Token features.
				for (int ftr = 0; ftr < getNumberOfFeatures(); ++ftr) {
					String ftrVal = example
							.getFeatureValueAsString(idxTkn, ftr);
					if (ftrVal.equals("0"))
						ftrVal = "O";
					ps.print(ftrVal + " ");
				}

				// Next line for the next token.
				ps.println();
			}

			// Seperate sentences by an empty line.
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
		return examples.get(idxExample).size();
	}

	/**
	 * Stub for an example.
	 * 
	 * This class does not store any additional information, it only stores
	 * references to an example.
	 * 
	 * @author eraldof
	 * 
	 */
	protected class Example implements DatasetExample {

		/**
		 * The index of this examples within the underlying dataset.
		 */
		protected int idxExample;

		/**
		 * Create an example stub that references the example in the given
		 * index.
		 * 
		 * @param idxExample
		 */
		protected Example(int idxExample) {
			this.idxExample = idxExample;
		}

		@Override
		public Dataset getDataset() {
			return Dataset.this;
		}

		@Override
		public String getID() {
			return exampleIDs.get(idxExample);
		}

		@Override
		public int size() {
			return examples.get(idxExample).size();
		}

		@Override
		public FeatureValueEncoding getFeatureEncoding() {
			return featureValueEncoding;
		}

		@Override
		public boolean containFeatureValue(int token, int feature, int value) {
			return (getFeatureValue(token, feature) == value);
		}

		@Override
		public void setFeatureValue(int token, int feature, int value) {
			examples.get(idxExample).get(token).set(feature, value);
		}

		@Override
		public void setFeatureValue(int token, String feature, String value)
				throws DatasetException {
			examples.get(idxExample)
					.get(token)
					.set(getFeatureIndex(feature),
							featureValueEncoding.putString(value));
		}

		@Override
		public void setFeatureValue(int token, int feature, String value)
				throws DatasetException {
			examples.get(idxExample).get(token)
					.set(feature, featureValueEncoding.putString(value));
		}

		@Override
		public int getFeatureValue(int token, int feature) {
			Integer val = examples.get(idxExample).get(token).get(feature);
			assert (val != null);
			return val;
		}

		@Override
		public String getFeatureValueAsString(int token, int feature) {
			return featureValueEncoding.getStringByCode(getFeatureValue(token,
					feature));
		}

		@Override
		public String getFeatureValueAsString(int token, String feature)
				throws DatasetException {
			return featureValueEncoding.getStringByCode(getFeatureValue(token,
					getFeatureIndex(feature)));
		}

		@Override
		public int getIndex() {
			return idxExample;
		}

		@Override
		protected Object clone() throws CloneNotSupportedException {
			return new Example(idxExample);
		}

		@Override
		public String toString() {
			StringBuilder builder = new StringBuilder();
			// Id.
			builder.append(getID());

			int size = size();
			int numFtrs = getNumberOfFeatures();
			for (int token = 0; token < size; ++token) {
				builder.append('\t' + getFeatureValueAsString(token, 0));
				for (int idxFtr = 1; idxFtr < numFtrs; ++idxFtr)
					builder.append(' ' + getFeatureValueAsString(token, idxFtr));
			}

			return builder.toString();
		}
	}

	/**
	 * Internal iterator representation.
	 * 
	 * WARNING! The current implementation prevents using more than one example
	 * objects, returned by the next method, at the same time. To do this, the
	 * client needs to clone the returned object, since the returned object is
	 * always the same. This behavior prevents memory leaks.
	 * 
	 * @author eraldof
	 * 
	 */
	private class DatasetIterator implements Iterator<DatasetExample> {

		private Example curExample;

		private DatasetIterator() {
			curExample = new Example(-1);
		}

		@Override
		public boolean hasNext() {
			return curExample.idxExample < getNumberOfExamples() - 1;
		}

		@Override
		public DatasetExample next() {
			++curExample.idxExample;
			return curExample;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
	}
}
