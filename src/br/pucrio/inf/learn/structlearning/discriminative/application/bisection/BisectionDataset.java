package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.SimpleExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent an author-papers dataset in column format (fixed number of
 * features).
 * 
 * Each author comprises a list of paper pairs. Authors are separated by blank
 * lines. A paper pair represents an edge in the graph whose nodes are the
 * candidate papers. Each edge is represented by fixed-length list of features.
 * 
 * The first line of a dataset must contain the list of feature labels. An
 * optional following line can list a subset of numerical features. All other
 * features are trated as cathegorical, i.e., each value present in the list of
 * edge features is considered a distinct value for the feature. All
 * cathegorical features are binarized.
 * 
 */
public class BisectionDataset implements Dataset {

	/**
	 * Loging object.
	 */
	private static final Log LOG = LogFactory.getLog(BisectionDataset.class);

	/**
	 * Regular expression pattern to parse spaces.
	 */
	private static final Pattern REGEX_SPACE = Pattern.compile("[ ]+");

	/**
	 * Map basic feature values (string) to codes (integer).
	 */
	private FeatureEncoding<String> basicEncoding;

	/**
	 * Encoding for derived features, i.e., features derived from templates that
	 * conjoin basic features.
	 */
	private FeatureEncoding<Feature> derivedEncoding;

	/**
	 * Array of input structures.
	 */
	private ExampleInputArray inputExamples;

	/**
	 * Array of golden output structures (correct prediction for the input
	 * structures).
	 */
	private BisectionOutput[] outputExamples;

	/**
	 * Feature templates to generate complex features from basic features.
	 */
	private BisectionTemplate[] templates;

	/**
	 * Labels of categorical basic features.
	 */
	private String[] categoricalFeatureLabels;

	/**
	 * Labels of numerical basic features.
	 */
	private String[] numericalFeatureLabels;

	/**
	 * Flags indicating which feature indexes (according to the dataset feature
	 * order) correspond to numerical features.
	 */
	private boolean[] numericalFeature;

	/**
	 * Index of the edge ID feature, which contains the indexes of the edge
	 * papers.
	 */
	private int idxFtrEdgeId;

	/**
	 * Index of the author ID feature.
	 */
	private int idxFtrAuthorId;

	/**
	 * Index of the paper 1 id feature.
	 */
	private int idxFtrPaper1Id;

	/**
	 * Index of the paper 2 id feature.
	 */
	private int idxFtrPaper2Id;

	/**
	 * Default constructor.
	 */
	public BisectionDataset() {
		this(new StringMapEncoding(), new MapEncoding<Feature>());
	}

	/**
	 * Create a dataset using the given feature-value encodings. One can use
	 * this constructor to create a dataset compatible with a previous loaded
	 * model, for instance.
	 * 
	 * @param basicEncoding
	 */
	public BisectionDataset(FeatureEncoding<String> basicEncoding,
			FeatureEncoding<Feature> explicitEnconding) {
		this.basicEncoding = basicEncoding;
		this.derivedEncoding = explicitEnconding;
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
	public BisectionDataset(String fileName) throws IOException,
			DatasetException {
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
	public BisectionDataset(InputStream is) throws IOException,
			DatasetException {
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
	public BisectionDataset(String fileName,
			FeatureEncoding<String> basicEncoding,
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
	public BisectionDataset(BisectionDataset sibling) {
		this.basicEncoding = sibling.basicEncoding;
		this.derivedEncoding = sibling.derivedEncoding;
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
	 * @throws DatasetException
	 */
	public void loadTemplates(String templatesFileName, boolean generateFeatures)
			throws IOException, DatasetException {
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
	 * @throws DatasetException
	 */
	public void loadTemplates(BufferedReader reader, boolean generateFeatures)
			throws IOException, DatasetException {
		LinkedList<BisectionTemplate> templatesList = new LinkedList<BisectionTemplate>();
		String line = skipBlanksAndComments(reader);
		while (line != null) {
			String[] ftrsStr = REGEX_SPACE.split(line);
			LinkedList<Integer> categoricalFeatureIndexes = new LinkedList<Integer>();
			LinkedList<Integer> numericalFeatureIndexes = new LinkedList<Integer>();
			for (int idx = 0; idx < ftrsStr.length; ++idx) {
				String ftrLabel = ftrsStr[idx];
				// Try to get the feature index from the categorical labels.
				int idxFtr = getCategoricalFeatureIndex(ftrLabel);
				if (idxFtr < 0) {
					// Not categorical. Try numerical labels.
					idxFtr = getNumericalFeatureIndex(ftrLabel);
					if (idxFtr < 0)
						// Not numerical. Error!
						throw new DatasetException(String.format(
								"Unknown feature label %s", ftrLabel));
					// Numerical feature.
					numericalFeatureIndexes.add(idxFtr);
				} else {
					// Categorical feature.
					categoricalFeatureIndexes.add(idxFtr);
				}
			}

			// Create template and add it to template list.
			templatesList.add(new BisectionTemplate(templatesList.size(),
					categoricalFeatureIndexes, numericalFeatureIndexes));

			// Read next line.
			line = skipBlanksAndComments(reader);
		}

		// Convert list to array.
		templates = templatesList.toArray(new BisectionTemplate[0]);

		if (generateFeatures)
			// Generate explicit features.
			generateFeatures();
	}

	/**
	 * Generate features for the current template partition.
	 */
	public void generateFeatures() {
		// List of feature codes used for every edge.
		LinkedList<Integer> ftrCodes = new LinkedList<Integer>();
		// List of feature values used for every edge.
		LinkedList<Double> ftrValues = new LinkedList<Double>();
		// Iterate over all examples.
		int numExs = inputExamples.getNumberExamples();
		
		inputExamples.loadInOrder();
		
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			// Current input structure.
			BisectionInput input = (BisectionInput) inputExamples.get(idxEx);

			// Allocate derived features matrix.
			input.allocFeatureArray();

			// Number of papers within the current input.
			int size = input.size();

			for (int paper1 = 0; paper1 < size; ++paper1) {
				for (int paper2 = 0; paper2 < size; ++paper2) {
					// Values of basic categorical features in the current edge.
					int[] basicCategoricalFeatures = input
							.getBasicCategoricalFeatures(paper1, paper2);
					if (basicCategoricalFeatures == null)
						// Inexistent edge. Skip it.
						continue;
					// Values of basic numerical features in the current edge.
					double[] basicNumericalFeatures = input
							.getBasicNumericalFeatures(paper1, paper2);

					// Clear auxiliary lists.
					ftrCodes.clear();
					ftrValues.clear();

					/*
					 * Instantiate edge features and add them to active features
					 * list.
					 */
					for (int idxTpl = 0; idxTpl < templates.length; ++idxTpl) {
						// Current template.
						BisectionTemplate tpl = templates[idxTpl];
						// Get temporary feature instance.
						Feature ftr = tpl.getInstance(input, paper1, paper2);
						// Lookup the feature in the encoding.
						int code = derivedEncoding.getCodeByValue(ftr);
						/*
						 * Instantiate a new feature, if it is not present in
						 * the encoding.
						 */
						if (code == FeatureEncoding.UNSEEN_VALUE_CODE) {
							try {
								code = derivedEncoding.put(ftr.clone());
							} catch (CloneNotSupportedException e) {
								LOG.error("Some feature is not cloneable", e);
							}
						}
						// Add feature code to the features list.
						ftrCodes.add(code);
						/*
						 * Compute derived feature value for this edge. This
						 * value is the multiplication of each numerical feature
						 * value in the edge.
						 */
						double val = 1d;
						for (int idxNumFtr : tpl.getNumericalFeatures())
							val *= basicNumericalFeatures[idxNumFtr];
						ftrValues.add(val);
					}

					// Set derived features codes and values for this edge.
					input.setFeatures(paper1, paper2, ftrCodes, ftrValues);
				}
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
	 * Return the index of the given categorical basic feature label. Return -1
	 * if there is no categorical feature labeled like this.
	 * 
	 * @param label
	 * @return
	 */
	private int getCategoricalFeatureIndex(String label) {
		for (int idx = 0; idx < categoricalFeatureLabels.length; ++idx)
			if (label.equals(categoricalFeatureLabels[idx]))
				return idx;
		return -1;
	}

	/**
	 * Return the index of the given numerical basic feature label. Return -1 if
	 * the is no numerical feature labeled like this.
	 * 
	 * @param label
	 * @return
	 */
	private int getNumericalFeatureIndex(String label) {
		for (int idx = 0; idx < numericalFeatureLabels.length; ++idx)
			if (label.equals(numericalFeatureLabels[idx]))
				return idx;
		return -1;
	}

	private int getIndex(String[] labels, String label) {
		int idx = 0;
		for (String l : labels) {
			if (label.equals(l.trim()))
				return idx;
			++idx;
		}
		return -1;
	}

	@Override
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		// Read feature labels in the first line of the file.
		String line = skipBlanksAndComments(reader);
		int eq = line.indexOf('=');
		int end = line.indexOf(']');
		String[] labels = line.substring(eq + 1, end).split(",");

		// Flags indicating which features are numerical.
		numericalFeature = new boolean[labels.length];

		// Read numerical feature labels in the second line.
		String numFtrsLine = reader.readLine();
		String[] labelsNumFtrs = null;
		if (numFtrsLine.trim().length() > 0) {
			eq = numFtrsLine.indexOf('=');
			end = numFtrsLine.indexOf(']');
			labelsNumFtrs = numFtrsLine.substring(eq + 1, end).split(",");
			// Set numerical feature flags.
			for (String numFtrLabel : labelsNumFtrs)
				numericalFeature[getIndex(labels, numFtrLabel.trim())] = true;
			// Skip first blank line.
			reader.readLine();
		} else
			labelsNumFtrs = new String[0];

		// Fill the numberical labels in the order of the original ftrs.
		numericalFeatureLabels = new String[labelsNumFtrs.length];
		int idxNumFtr = 0;
		for (int idxFtr = 0; idxFtr < labels.length; ++idxFtr)
			if (numericalFeature[idxFtr])
				numericalFeatureLabels[idxNumFtr++] = labels[idxFtr].trim();

		/*
		 * Fill the categorical feature labels in the order of the original
		 * ftrs. The number of categorical features is equal to the total number
		 * of labels minus the number of numerical features minus 1 (due to the
		 * gold-standard feature).
		 */
		categoricalFeatureLabels = new String[labels.length
				- labelsNumFtrs.length - 1];
		int idxCatFtr = 0;
		for (int idxFtr = 0; idxFtr < labels.length - 1; ++idxFtr)
			if (!numericalFeature[idxFtr])
				categoricalFeatureLabels[idxCatFtr++] = labels[idxFtr].trim();

		// Get index of the feature with edge id.
		idxFtrEdgeId = getIndex(labels, "id");
		// Get index of the author ID feature.
		idxFtrAuthorId = getIndex(labels, "authorId");
		// Get index of the paper1 id feature.
		idxFtrPaper1Id = getIndex(labels, "paperId1");
		// Get index of the paper2 id feature.
		idxFtrPaper2Id = getIndex(labels, "paperId2");

		// Examples.
		List<BisectionInput> inputList = new LinkedList<BisectionInput>();
		List<BisectionOutput> outputList = new LinkedList<BisectionOutput>();
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
		
		for (BisectionInput bisectionInput : inputList) {
			inputExamples.put(bisectionInput);
		}
		
		
		outputExamples = outputList.toArray(new BisectionOutput[0]);

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
			Collection<BisectionInput> exampleInputs,
			Collection<BisectionOutput> exampleOutputs)
			throws DatasetException, NumberFormatException, IOException {

		// Categorical features of edges.
		LinkedList<LinkedList<Integer>> edgesCatFtrCodes = new LinkedList<LinkedList<Integer>>();
		LinkedList<LinkedList<Double>> edgesNumFtrValues = new LinkedList<LinkedList<Double>>();

		/*
		 * Confirmed and deleted papers for this example. These two lists will
		 * comprise the output structure.
		 */
		HashSet<Integer> confirmedPapers = new HashSet<Integer>();

		// Papers ids.
		ArrayList<Long> papersIds = new ArrayList<Long>();

		String line;
		// Guarantee that every edge has the same author ID.
		long authorId = -1;
		long prevAuthorId = Long.MAX_VALUE;
		// Size of input structure is obtained from the papers ids.
		int size = 0;
		while ((line = reader.readLine()) != null) {
			// Trim read line.
			line = line.trim();
			if (line.length() == 0)
				// Stop on blank lines.
				break;
			if (line.startsWith("#"))
				// Comment line.
				continue;

			// Feature values in string format.
			String[] ftrStrs = REGEX_SPACE.split(line);

			// Get edge ID feature and extract papers indexes from it.
			String edgeId = ftrStrs[idxFtrEdgeId];
			String[] papersStr = edgeId.split(">");
			int paper1 = Integer.parseInt(papersStr[0]);
			int paper2 = Integer.parseInt(papersStr[1]);

			// Update number of papers.
			size = Math.max(paper1 + 1, size);
			size = Math.max(paper2 + 1, size);

			// Adjust array of papers ids to the correct size.
			int piSize = papersIds.size();
			if (piSize < size)
				for (int i = 0; i < size - piSize; ++i)
					papersIds.add(null);

			// Set ids for both papers of this edge.
			Long p1Id = papersIds.get(paper1);
			if (p1Id == null)
				papersIds.set(paper1, Long.parseLong(ftrStrs[idxFtrPaper1Id]));
			else if (p1Id != Long.parseLong(ftrStrs[idxFtrPaper1Id]))
				throw new DatasetException(String.format(
						"Edge (%d,%d) presents inconsistent paper1 id.",
						paper1, paper2));
			Long p2Id = papersIds.get(paper2);
			if (p2Id == null)
				papersIds.set(paper2, Long.parseLong(ftrStrs[idxFtrPaper2Id]));
			else if (p2Id != Long.parseLong(ftrStrs[idxFtrPaper2Id]))
				throw new DatasetException(String.format(
						"Edge (%d,%d) presents inconsistent paper2 id.",
						paper1, paper2));

			// Get author ID and verify that it is the same of previous edge.
			authorId = Long.parseLong(ftrStrs[idxFtrAuthorId]);
			if (prevAuthorId != Long.MAX_VALUE && prevAuthorId != authorId)
				System.err
						.println("Different author IDs within the same example!");
			prevAuthorId = authorId;

			// List of feature codes (categorical).
			LinkedList<Integer> ftrCodes = new LinkedList<Integer>();
			// List of feature values (numerical).
			LinkedList<Double> ftrValues = new LinkedList<Double>();
			// Add them already to the lists of edges.
			edgesCatFtrCodes.add(ftrCodes);
			edgesNumFtrValues.add(ftrValues);

			// Add papers ids to the list of categorical feature codes.
			ftrCodes.add(paper1);
			ftrCodes.add(paper2);

			// Encode categorical edge features and parse numerical ones.
			for (int idxFtr = 0; idxFtr < ftrStrs.length - 1; ++idxFtr) {
				String str = ftrStrs[idxFtr];
				if (numericalFeature[idxFtr])
					// Numerical feature.
					ftrValues.add(Double.parseDouble(str));
				else
					// Categorical feature.
					ftrCodes.add(basicEncoding.put(new String(str)));
			}

			// The last value is the correct feature (Y or N).
			String areConfirmedPapers = ftrStrs[ftrStrs.length - 1];

			/*
			 * Add paper to the confirmed set, whenever appropriate, and check
			 * the consistency of this information.
			 */
			if (paper1 == 0) {
				// Edge from the artificial DELETED paper.
				if ("N".equals(areConfirmedPapers))
					confirmedPapers.add(paper2);
				else if (confirmedPapers.contains(paper2))
					throw new DatasetException(String.format(
							"For author %d, edge (%d,%d) is indicated "
									+ "as correct, but this is "
									+ "inconsistent with previous edges.",
							authorId, paper1, paper2));
			} else if (paper1 == 1) {
				// Edge from the artificial CONFIRMED paper.
				if ("Y".equals(areConfirmedPapers))
					confirmedPapers.add(paper2);
				else if (confirmedPapers.contains(paper2))
					throw new DatasetException(String.format(
							"For author %d, edge (%d,%d) is indicated "
									+ "as incorrect, but this is "
									+ "inconsistent with previous edges.",
							authorId, paper1, paper2));
			} else if (paper2 == 0) {
				// Edge to the artificial DELETED paper.
				if ("N".equals(areConfirmedPapers))
					confirmedPapers.add(paper1);
				else if (confirmedPapers.contains(paper1))
					throw new DatasetException(String.format(
							"For author %d, edge (%d,%d) is indicated "
									+ "as correct, but this is "
									+ "inconsistent with previous edges.",
							authorId, paper1, paper2));
			} else if (paper2 == 1) {
				// Edge to the artificial CONFIRMED paper.
				if ("Y".equals(areConfirmedPapers))
					confirmedPapers.add(paper1);
				else if (confirmedPapers.contains(paper1))
					throw new DatasetException(String.format(
							"For author %d, edge (%d,%d) is indicated "
									+ "as incorrect, but this is "
									+ "inconsistent with previous edges.",
							authorId, paper1, paper2));
			}
		}

		if (edgesCatFtrCodes.size() == 0)
			return line != null;

		// Create new input struct with the parsed features.
		BisectionInput input = new BisectionInput(size, authorId, papersIds,
				edgesCatFtrCodes, edgesNumFtrValues);

		// Create confirmed array.
		boolean[] confirmed = new boolean[size];
		// Paper 1 is the artificial CONFIRMED paper.
		confirmed[1] = true;
		for (int paper : confirmedPapers)
			confirmed[paper] = true;

		// Allocate the output structure.
		BisectionOutput output = new BisectionOutput(confirmed);

		// Update lists of input and output structures.
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
	public BisectionOutput[] getOutputs() {
		return outputExamples;
	}

	@Override
	public BisectionInput getInput(int index) {
		throw new NotImplementedException();
	}

	@Override
	public BisectionOutput getOutput(int index) {
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
		return derivedEncoding;
	}

	/**
	 * Save the given predicted information in CSV format (KDD Cup 2013 format).
	 * 
	 * @param testOutFilename
	 * @param predicteds
	 * @throws FileNotFoundException
	 */
	public void save(String testOutFilename, BisectionOutput[] predicteds)
			throws FileNotFoundException {
		PrintStream ps = new PrintStream(testOutFilename);
		save(ps, predicteds);
		ps.close();
	}

	public void save(PrintStream ps, BisectionOutput[] predicteds) {
		int numExs = getNumberOfExamples();
		inputExamples.loadInOrder();
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			BisectionInput in = (BisectionInput) inputExamples.get(idxEx);
			BisectionOutput out = predicteds[idxEx];
			ps.print(in.getAuthorId() + ",");
			int size = out.size();
			for (int idxPaper = 0; idxPaper < size; ++idxPaper) {
				int paper = out.weightedPapers[idxPaper].paper;
				if (paper == 0 || paper == 1)
					// Skip artificial papers.
					continue;
				ps.print(in.getPaperId(paper) + " ");
			}
			ps.println();
		}
	}
}
