package br.pucrio.inf.learn.structlearning.generative.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Vector;

import org.apache.log4j.Logger;

/**
 * A dataset class that does not load all the examples in memory. It loads
 * example by example. So, this class implements basically the methods necessary
 * for the iterable interface.
 * 
 * @author eraldof
 * 
 */
public class MemoryEfficientDataset extends Corpus {

	private static Logger logger = Logger
			.getLogger(MemoryEfficientDataset.class);

	protected int lastExampleSize;

	/**
	 * List of file names to be loaded.
	 */
	protected String[] inputFileNames;

	public MemoryEfficientDataset() {
		super();
	}

	public MemoryEfficientDataset(FeatureValueEncoding featureValueEncoding) {
		super(featureValueEncoding);
	}

	public MemoryEfficientDataset(String fileName) throws IOException,
			DatasetException {
		super();
		featureValueEncoding = new FeatureValueEncoding();
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>(1);
		examples.add(new Vector<Vector<Integer>>());
		exampleIDs = new Vector<String>(1);
		exampleIDs.add(new String());
		inputFileNames = new String[1];
		inputFileNames[0] = fileName;
	}

	public MemoryEfficientDataset(String[] fileNames) {
		featureValueEncoding = new FeatureValueEncoding();
		featureLabels = new Vector<String>();
		examples = new Vector<Vector<Vector<Integer>>>(1);
		examples.add(new Vector<Vector<Integer>>());
		exampleIDs = new Vector<String>(1);
		exampleIDs.add(new String());
		inputFileNames = fileNames.clone();
	}

	public MemoryEfficientDataset(String fileName,
			FeatureValueEncoding featureValueEncoding) throws IOException,
			DatasetException {
		super(fileName, featureValueEncoding);
	}

	@Override
	public Iterator<DatasetExample> iterator() {
		try {
			return new MemoryEfficientDatasetIterator();
		} catch (IOException e) {
			logger.error("Reading the input files", e);
			return null;
		}
	}

	public boolean parseExample(String buff) throws DatasetException {
		// Split tokens.
		String tokens[] = buff.split("[\t]");

		if (tokens.length == 0)
			return false;

		// The first field is the sentence id.
		String id = tokens[0];

		if (id.trim().length() == 0)
			return false;

		Vector<Vector<Integer>> exampleData = examples.get(0);

		// Adjust the vector to the size of the current example (if it is
		// needed).
		int prevSize = exampleData.size();
		if (prevSize < tokens.length - 1) {
			exampleData.setSize(tokens.length - 1);
			for (int idx = prevSize; idx < exampleData.size(); ++idx) {
				exampleData
						.set(idx, new Vector<Integer>(getNumberOfFeatures()));
				exampleData.get(idx).setSize(getNumberOfFeatures());
			}
		}

		lastExampleSize = exampleData.size();

		for (int idxTkn = 1; idxTkn < tokens.length; ++idxTkn) {
			String token = tokens[idxTkn];

			// Split the token into its features.
			String[] features = token.split("[ ]");
			if (features.length != getNumberOfFeatures())
				throw new DatasetException(
						"Incorrect number of features on the following example:\n"
								+ buff);

			// Encode the feature values.
			Vector<Integer> featuresAsVector = exampleData.get(idxTkn - 1);
			for (int idxFtr = 0; idxFtr < getNumberOfFeatures(); ++idxFtr)
				featuresAsVector.set(idxFtr,
						featureValueEncoding.putString(features[idxFtr]));
		}

		// Store the loaded example.
		exampleIDs.set(0, id);

		return true;
	}

	@Override
	public DatasetExample getExample(int index) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void add(Corpus dataset) throws DatasetException {
		throw new UnsupportedOperationException();
	}

	@Override
	public int getNumberOfExamples() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DatasetExample addExample(String id,
			Collection<? extends Collection<Integer>> exampleFeatures)
			throws DatasetException {
		throw new UnsupportedOperationException();
	}

	@Override
	public DatasetExample addExampleAsString(String id,
			Collection<? extends Collection<String>> exampleFeatureLabels)
			throws DatasetException {
		throw new UnsupportedOperationException();
	}

	@Override
	public void load(String fileName) throws IOException, DatasetException {
		throw new UnsupportedOperationException();
	}

	@Override
	public void load(InputStream is) throws IOException, DatasetException {
		throw new UnsupportedOperationException();
	}

	@Override
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		throw new UnsupportedOperationException();
	}

	@Override
	public void save(String fileName) throws IOException {
		throw new UnsupportedOperationException();
	}

	@Override
	public void save(PrintStream ps) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void saveAsCoNLL(String fileName) throws IOException {
		throw new UnsupportedOperationException();
	}

	@Override
	public void saveAsCoNLL(PrintStream ps) {
		throw new UnsupportedOperationException();
	}

	@Override
	public int getNumberOfTokens(int idxExample) {
		throw new UnsupportedOperationException();
	}

	private class Example extends Corpus.Example {

		protected int size;

		protected Example(int idxExample) {
			super(idxExample);
		}

		protected void setSize(int newSize) {
			size = newSize;
		}

		public int size() {
			return size;
		}
	}

	/**
	 * Iterate over the file list of the dataset.
	 * 
	 * @author eraldof
	 * 
	 */
	private class MemoryEfficientDatasetIterator implements
			Iterator<DatasetExample> {

		/**
		 * Last loaded line. Always load a line in advance, i.e., before the
		 * <code>next</code> method call.
		 */
		private String lastLine;

		/**
		 * Use only one Example object. WARNING! This prevents using multiple
		 * Example references returned by the iterator.
		 */
		private Example curExample;

		/**
		 * Index within the file names array of the file being processed.
		 */
		private int curFileIndex;

		/**
		 * Current file reader, i.e., the file reader that is being loaded.
		 */
		private BufferedReader curFileReader;

		public MemoryEfficientDatasetIterator() throws IOException {
			curExample = new Example(0);
			curFileIndex = 0;
			curFileReader = new BufferedReader(new FileReader(
					inputFileNames[curFileIndex]));
			lastLine = skipBlanksAndComments(curFileReader);
		}

		/**
		 * Deal with the multiple files (skiping one when it is done).
		 * 
		 * @param reader
		 * @return
		 * @throws IOException
		 */
		protected String skipBlanksAndComments(BufferedReader reader)
				throws IOException {
			// If there is no remaining file.
			if (curFileIndex >= inputFileNames.length)
				return null;

			String buff;
			while ((buff = MemoryEfficientDataset.this
					.skipBlanksAndComments(curFileReader)) == null) {
				// Close the previous file.
				curFileReader.close();

				System.out.print(".");

				// Go to the next file. Stop, if there is no one left.
				++curFileIndex;
				if (curFileIndex >= inputFileNames.length)
					return null;

				// Open the next file.
				curFileReader = new BufferedReader(new FileReader(
						inputFileNames[curFileIndex]));
			}

			return buff;
		}

		@Override
		public boolean hasNext() {
			return lastLine != null;
		}

		@Override
		public DatasetExample next() {
			if (lastLine == null)
				throw new NoSuchElementException("No element left");

			try {
				// Parse the previous loaded line.
				parseExample(lastLine);

				// Read the next line.
				lastLine = skipBlanksAndComments(curFileReader);
				
				curExample.setSize(lastExampleSize);

				// Return the stub for the parsed example.
				return curExample;

			} catch (DatasetException e) {
				throw new NoSuchElementException(e.getMessage());
			} catch (IOException e) {
				throw new NoSuchElementException(e.getMessage());
			}
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
	}
}
