package br.pucrio.inf.learn.structlearning.discriminative.application.rank;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.FeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.SimpleExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a query-items dataset in column format.
 * 
 * Provide methods for manipulating the dataset. Some operations are: load
 * examples from file, create new features, remove features, get feature values,
 * get a complete example, change feature values.
 * 
 * The feature values are stored as integer. We use a feature-value mapping to
 * encode the string values.
 * 
 */
public class RankDataset implements Dataset {

	/**
	 * Loging object.
	 */
	private static final Log LOG = LogFactory.getLog(RankDataset.class);

	/**
	 * Regular expression pattern to parse spaces.
	 */
	protected static final Pattern REGEX_SPACE = Pattern.compile("[ ]");

	/**
	 * Map basic feature values (string) to codes (integer).
	 */
	protected FeatureEncoding<String> basicEncoding;

	/**
	 * Encoding for explicit features, i.e., features created from templates by
	 * conjoining basic features.
	 */
	protected FeatureEncoding<Feature> explicitEncoding;

	/**
	 * Array of input structures.
	 */
	protected ExampleInputArray inputExamples;

	/**
	 * Array of golden output structures (correct prediction for the input
	 * structures).
	 */
	protected RankOutput[] outputExamples;

	/**
	 * Indicate whether the output structures in this dataset have golden
	 * information.
	 */
	protected boolean hasGoldenOutput;

	/**
	 * Feature templates to generate complex features from basic features.
	 */
	protected FeatureTemplate[] templates;

	/**
	 * Labels of this dataset columns (basic features).
	 */
	protected String[] featureLabels;

	/**
	 * Default constructor.
	 */
	public RankDataset() {
		this(new StringMapEncoding(), new MapEncoding<Feature>());
	}

	/**
	 * Create a dataset using the given feature-value encodings. One can use
	 * this constructor to create a dataset compatible with a previous loaded
	 * model, for instance.
	 * 
	 * @param basicEncoding
	 */
	public RankDataset(FeatureEncoding<String> basicEncoding,
			FeatureEncoding<Feature> explicitEnconding) {
		this.basicEncoding = basicEncoding;
		this.explicitEncoding = explicitEnconding;
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
	public RankDataset(String fileName) throws IOException, DatasetException {
		this(new StringMapEncoding(), new MapEncoding<Feature>());
		load(fileName);
	}

	/**
	 * Load the dataset from a <code>InputStream</code>.
	 * 
	 * @param is
	 * @throws IOException
	 * @throws DatasetException
	 */
	public RankDataset(InputStream is) throws IOException, DatasetException {
		this(new StringMapEncoding(), new MapEncoding<Feature>());
		load(is);
	}

	/**
	 * Load the dataset from the given file and use the given feature-value
	 * encodings. One can use this constructor to load a dataset compatible with
	 * a previous loaded model, for instance.
	 * 
	 * @param fileName
	 *            name and path of a file.
	 * @param basicEncoding
	 *            use a determined feature values encoding.
	 * 
	 * @throws IOException
	 *             if occurs some problem reading the file.
	 * @throws DatasetException
	 *             if the file contains invalid data.
	 */
	public RankDataset(String fileName, FeatureEncoding<String> basicEncoding,
			FeatureEncoding<Feature> explicitEncoding) throws IOException,
			DatasetException {
		this(basicEncoding, explicitEncoding);
		load(fileName);
	}

	/**
	 * Create a sibling dataset.
	 * 
	 * @param sibling
	 */
	public RankDataset(RankDataset sibling) {
		this.basicEncoding = sibling.basicEncoding;
		this.explicitEncoding = sibling.explicitEncoding;
		this.templates = sibling.templates;
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
	 * Load templates from the given file and, optionally, generate explicit
	 * features.
	 * 
	 * @param templatesFileName
	 * @param generateFeatures
	 * @throws IOException
	 */
	public void loadTemplates(String templatesFileName, boolean generateFeatures)
			throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		loadTemplates(reader, generateFeatures);
		reader.close();
	}

	/**
	 * Load templates from the given reader and, optionally, generate explicit
	 * features.
	 * 
	 * @param reader
	 * @param generateFeatures
	 * @throws IOException
	 */
	public void loadTemplates(BufferedReader reader, boolean generateFeatures)
			throws IOException {
		LinkedList<FeatureTemplate> templatesList = new LinkedList<FeatureTemplate>();
		String line = skipBlanksAndComments(reader);
		while (line != null) {
			String[] ftrsStr = REGEX_SPACE.split(line);
			int[] ftrs = new int[ftrsStr.length];
			for (int idx = 0; idx < ftrs.length; ++idx)
				ftrs[idx] = getFeatureIndex(ftrsStr[idx]);
			templatesList.add(new RankTemplate(templatesList.size(), ftrs));
			// Read next line.
			line = skipBlanksAndComments(reader);
		}

		// Convert list to array.
		templates = templatesList.toArray(new FeatureTemplate[0]);

		if (generateFeatures)
			// Generate explicit features.
			generateFeatures();
	}

	/**
	 * Generate features for the current template partition.
	 */
	public void generateFeatures() {
		LinkedList<Integer> ftrs = new LinkedList<Integer>();
		FeatureTemplate[] tpls = templates;
		int numExs = inputExamples.getNumberExamples();
		
		inputExamples.loadInOrder();
		
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			// Current input structure.
			RankInput input = (RankInput) inputExamples.get(idxEx);

			// Allocate explicit features matrix.
			input.allocFeatureArray();

			// Number of tokens within the current input.
			int size = input.size();

			for (int item = 0; item < size; ++item) {

				// Clear previous used list of features.
				ftrs.clear();

				/*
				 * Instantiate edge features and add them to active features
				 * list.
				 */
				for (int idxTpl = 0; idxTpl < tpls.length; ++idxTpl) {
					// Current template.
					FeatureTemplate tpl = tpls[idxTpl];
					// Get temporary feature instance.
					Feature ftr = tpl.getInstance(input, item);
					// Lookup the feature in the encoding.
					int code = explicitEncoding.getCodeByValue(ftr);
					/*
					 * Instantiate a new feature, if it is not present in the
					 * encoding.
					 */
					if (code == FeatureEncoding.UNSEEN_VALUE_CODE)
						code = explicitEncoding.put(tpl
								.newInstance(input, item));
					// Add feature code to active features list.
					ftrs.add(code);
				}

				// Set feature vector of this input.
				input.setFeatures(item, ftrs);
			}

			// Progess report.
			if ((idxEx + 1) % 100 == 0) {
				System.out.print('.');
				System.out.flush();
			}
		}

		System.out.println();
		System.out.flush();
	}

	/**
	 * Return the index of the given feature label. Return -1 if there is no
	 * feature labeled like this.
	 * 
	 * @param label
	 * @return
	 */
	private int getFeatureIndex(String label) {
		for (int idx = 0; idx < featureLabels.length; ++idx)
			if (label.equals(featureLabels[idx]))
				return idx;
		return -1;
	}

	@Override
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		// Read feature labels in the first line of the file.
		String line = reader.readLine();
		reader.readLine();
		int eq = line.indexOf('=');
		int end = line.indexOf(']');
		String[] labels = line.substring(eq + 1, end).split(",");
		// Do not include the last feature (relevant).
		featureLabels = new String[labels.length - 1];
		for (int i = 0; i < labels.length - 1; ++i)
			featureLabels[i] = labels[i].trim();

		// Examples.
		List<RankInput> inputList = new LinkedList<RankInput>();
		List<RankOutput> outputList = new LinkedList<RankOutput>();
		int numExs = 0;
		while (parseExample(reader, inputList, outputList)) {
			++numExs;
			if ((numExs + 1) % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}
		System.out.println();

		// Convert the list of examples (input and output) to arrays.
		inputExamples = new SimpleExampleInputArray(inputList.size());
		
		for (RankInput rankInput : inputList) {
			inputExamples.put(rankInput);
		}
		
		outputExamples = outputList.toArray(new RankOutput[0]);

		LOG.info("Read " + numExs + " examples.");
	}

	/**
	 * Parse an example in the next lines of the given reader. An example
	 * comprises a list of items. Each item is represented in one line. An item
	 * comprises a list of feature values.
	 * 
	 * @param reader
	 * @param exampleInputs
	 * @param exampleOutputs
	 * @return
	 * @throws DatasetException
	 * @throws NumberFormatException
	 * @throws IOException
	 */
	private boolean parseExample(BufferedReader reader,
			Collection<RankInput> exampleInputs,
			Collection<RankOutput> exampleOutputs) throws DatasetException,
			NumberFormatException, IOException {
		// List of items of this query. Each item is a list of features.
		LinkedList<LinkedList<Integer>> items = new LinkedList<LinkedList<Integer>>();

		// Relevant and irrelevant items for this query.
		LinkedList<Integer> relevantItems = new LinkedList<Integer>();
		LinkedList<Integer> irrelevantItems = new LinkedList<Integer>();

		// Read next line.
		String line;
		long prevQueryId = Long.MAX_VALUE;
		while ((line = reader.readLine()) != null) {
			// Trim read line.
			line = line.trim();
			if (line.length() == 0)
				// Stop on blank lines.
				break;

			// Split edge in feature values.
			String[] ftrValues = REGEX_SPACE.split(line);

			// Head and dependent tokens indexes.
			long queryId = Long.parseLong(ftrValues[0]);
			if (prevQueryId != Long.MAX_VALUE && prevQueryId != queryId)
				System.err
						.println("Different query IDs within the same example!");
			prevQueryId = queryId;

			// List of feature codes.
			LinkedList<Integer> ftrCodes = new LinkedList<Integer>();
			// Add it already to the list of items.
			items.add(ftrCodes);

			// Encode the edge features and add the codes to the list.
			for (int idxFtr = 0; idxFtr < ftrValues.length - 1; ++idxFtr) {
				String str = ftrValues[idxFtr];
				int code = basicEncoding.put(new String(str));
				ftrCodes.add(code);
			}

			// Index of the current item.
			int item = items.size() - 1;

			// The last value is the relevant feature (1 or -1).
			String isRelevant = ftrValues[ftrValues.length - 1];
			if (isRelevant.equals("1")) {

				/*
				 * Add the index of the current item to the list of relevant
				 * items.
				 */
				relevantItems.add(item);

			} else if (isRelevant.equals("-1")) {

				/*
				 * Add the index of the current item to the list of irrelevant
				 * items.
				 */
				irrelevantItems.add(item);

			} else {
				// Invalid feature value.
				throw new DatasetException(
						"Last feature value must be 1 or -1 to indicate "
								+ "whether the item is relevant or not. "
								+ "However, for item " + item
								+ " this feature value is " + isRelevant);
			}
		}

		if (items.size() == 0)
			return line != null;

		/*
		 * Create a new string to store the input id to avoid memory leaks,
		 * since the id string keeps a reference to the line string.
		 */
		RankInput input = new RankInput(prevQueryId, items);

		// Allocate the output structure.
		RankOutput output = new RankOutput(relevantItems, irrelevantItems);

		exampleInputs.add(input);
		exampleOutputs.add(output);

		// Return true if there are more lines.
		return line != null;
	}

	/**
	 * Skip blank lines and lines starting by the comment character #.
	 * 
	 * @param reader
	 * @return
	 * @throws IOException
	 */
	private String skipBlanksAndComments(BufferedReader reader)
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

	public FeatureEncoding<String> getBasicFeatureEncoding() {
		return basicEncoding;
	}

	@Override
	public int getNumberOfExamples() {
		return inputExamples.getNumberExamples();
	}

	@Override
	public ExampleInputArray getInputs() {
		return inputExamples;
	}

	@Override
	public RankOutput[] getOutputs() {
		return outputExamples;
	}

	@Override
	public RankInput getInput(int index) {
		throw new NotImplementedException();
//		return inputExamples[index];
	}

	@Override
	public RankOutput getOutput(int index) {
		return outputExamples[index];
	}

	@Override
	public void save(String fileName) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public boolean isTraining() {
		throw new NotImplementedException();
	}

	@Override
	public void save(BufferedWriter writer) throws IOException,
			DatasetException {
		throw new NotImplementedException();
	}

	@Override
	public void save(OutputStream os) throws IOException, DatasetException {
		throw new NotImplementedException();
	}

	public FeatureEncoding<Feature> getExplicitEncoding() {
		return explicitEncoding;
	}

	public void save(String testOutFilename, RankOutput[] predicteds)
			throws FileNotFoundException {
		PrintStream ps = new PrintStream(testOutFilename);
		save(ps, predicteds);
		ps.close();
	}

	public void save(PrintStream ps, RankOutput[] predicteds) {
		int numExs = getNumberOfExamples();
		
		inputExamples.loadInOrder();
		
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			RankInput in = (RankInput) inputExamples.get(idxEx);
			RankOutput out = predicteds[idxEx];
			ps.print(in.getQueryId() + ",");
			int numItems = out.size();
			for (int idxItem = 0; idxItem < numItems; ++idxItem) {
				int item = out.weightedItems[idxItem].item;
				ps.print(basicEncoding.getValueByCode(in.getBasicFeatures(item)[1])
						+ " ");
			}
			ps.println();
		}
	}
}
