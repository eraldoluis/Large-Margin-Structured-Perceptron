package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.CacheExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a dataset with dependency parsing examples. Each example consists
 * of a sentence and is represented by a <code>DPGSInput</code> object. An
 * example consists of a set of factors (grandparent and siblings). Each factor
 * is represented by a fixed number of features (column-based representation,
 * also commonly used in CoNLL shared tasks). This class is useful for
 * template-based models.
 * 
 * @author eraldo
 * 
 */
public class DPGSDataset implements Dataset {

	/**
	 * Logging object.
	 */
	private static Log LOG = LogFactory.getLog(DPGSDataset.class);

	/**
	 * Regular expression pattern to parse spaces.
	 */
	protected final Pattern REGEX_SPACE = Pattern.compile("[ ]");

	/**
	 * Regular expression pattern to parse factor identification and capture its
	 * parameters.
	 */
	protected final Pattern REGEX_ID = Pattern
			.compile("^(E|G|S|LEN)\\((\\d+),(\\d+),(\\d+)\\)$");

	/**
	 * Encoding for basic textual features (column-format features).
	 */
	protected FeatureEncoding<String> basicEncoding;

	/**
	 * Basic feature labels for edge factors.
	 */
	protected String[] featureLabelsEdge;

	/**
	 * Basic feature labels for grandparent factors.
	 */
	protected String[] featureLabelsGrandparent;

	/**
	 * Basic feature labels for siblings factors.
	 */
	protected String[] featureLabelsSiblings;

	/**
	 * Multi-valued edge features.
	 */
	protected Set<String> multiValuedEdgeFeatures;

	/**
	 * Multi-valued grandparent features.
	 */
	protected Set<String> multiValuedGrandparentFeatures;

	/**
	 * Multi-valued siblings features.
	 */
	protected Set<String> multiValuedSiblingsFeatures;

	/**
	 * Input sentences.
	 */
	protected ExampleInputArray inputs;

	/**
	 * Output structures.
	 */
	protected DPGSOutput[] outputs;

	/**
	 * Indicate if this dataset is a training set.
	 */
	protected boolean training;

	/**
	 * Length of the longest sequence in this dataset.
	 */
	protected int maxNumberOfTokens;

	/**
	 * Separator for multi-valued features.
	 */
	protected String separatorFeatureValues;

	/**
	 * Create an empty dataset with the given multi-valued basic features for
	 * grandparent and siblings factors.
	 * 
	 * @param multiValuedGrandparentFeatures
	 * @param multiValuedSiblingsFeatures
	 * @param separatorFeatureValues
	 */
	public DPGSDataset(String[] multiValuedEdgeFeatures,
			String[] multiValuedGrandparentFeatures,
			String[] multiValuedSiblingsFeatures, String separatorFeatureValues) {
		// Basic feature encoding.
		this.basicEncoding = new StringMapEncoding();
		// Multi-values EDGE features.
		this.multiValuedEdgeFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedEdgeFeatures.length; ++idxFtr)
			this.multiValuedEdgeFeatures.add(multiValuedEdgeFeatures[idxFtr]);
		// Multi-values GRANDPARENT features.
		this.multiValuedGrandparentFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedGrandparentFeatures.length; ++idxFtr)
			this.multiValuedGrandparentFeatures
					.add(multiValuedGrandparentFeatures[idxFtr]);
		// Multi-values SIBLINGS features.
		this.multiValuedSiblingsFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedSiblingsFeatures.length; ++idxFtr)
			this.multiValuedSiblingsFeatures
					.add(multiValuedSiblingsFeatures[idxFtr]);
		// Multi-values separator.
		this.separatorFeatureValues = separatorFeatureValues;
	}

	/**
	 * Create an empty dataset with the given multi-valued basic features and
	 * the given basic feature encoding.
	 * 
	 * @param multiValuedGrandparentFeatures
	 * @param multiValuedSiblingsFeatures
	 * @param separatorFeatureValues
	 * @param basicEncoding
	 */
	public DPGSDataset(String[] multiValuedEdgeFeatures,
			String[] multiValuedGrandparentFeatures,
			String[] multiValuedSiblingsFeatures,
			String separatorFeatureValues, FeatureEncoding<String> basicEncoding) {
		// Basic feature encoding.
		this.basicEncoding = basicEncoding;
		// Multi-values EDGE features.
		this.multiValuedEdgeFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedEdgeFeatures.length; ++idxFtr)
			this.multiValuedEdgeFeatures.add(multiValuedEdgeFeatures[idxFtr]);
		// Multi-values GRANDPARENT features.
		this.multiValuedGrandparentFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedGrandparentFeatures.length; ++idxFtr)
			this.multiValuedGrandparentFeatures
					.add(multiValuedGrandparentFeatures[idxFtr]);
		// Multi-values SIBLINGS features.
		this.multiValuedSiblingsFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedSiblingsFeatures.length; ++idxFtr)
			this.multiValuedSiblingsFeatures
					.add(multiValuedSiblingsFeatures[idxFtr]);
		// Multi-values separator.
		this.separatorFeatureValues = separatorFeatureValues;
	}

	/**
	 * Create a new dataset using the same encoding and other underlying data
	 * structures of the given dataset.
	 * 
	 * For most use cases, the underlying data structures within the sibling
	 * dataset should be kept unchanged after creating this new dataset.
	 * 
	 * @param dataset
	 */
	public DPGSDataset(DPGSDataset dataset) {
		this.basicEncoding = dataset.basicEncoding;
		this.multiValuedEdgeFeatures = dataset.multiValuedEdgeFeatures;
		this.multiValuedGrandparentFeatures = dataset.multiValuedGrandparentFeatures;
		this.multiValuedSiblingsFeatures = dataset.multiValuedSiblingsFeatures;
		this.separatorFeatureValues = dataset.separatorFeatureValues;
	}

	/**
	 * @return the number of features (columns) in this corpus.
	 */
	public int getNumberOfGrandparentFeatures() {
		return featureLabelsGrandparent.length;
	}

	public ExampleInputArray getDPGSInputArray() {
		return inputs;
	}

	/**
	 * Return array of input structures (sentence factors).
	 * 
	 * @return
	 */
	public ExampleInputArray getInputs() {
		return inputs;
	}

	/**
	 * Return array of output structures (dependency parse, grandparent
	 * structure and siblings structure)
	 * 
	 * @return
	 */
	public DPGSOutput[] getOutputs() {
		return outputs;
	}

	/**
	 * Return the input in the given index.
	 * 
	 * @param index
	 * @return
	 */
	public DPGSInput getInput(int index) {
		throw new NotImplementedException();
		// return inputs[index];
	}

	/**
	 * Return the output in the given index.
	 * 
	 * @param index
	 * @return
	 */
	public DPGSOutput getOutput(int index) {
		return outputs[index];
	}

	/**
	 * Return number of examples.
	 * 
	 * @return
	 */
	public int getNumberOfExamples() {
		return inputs.getNumberExamples();
	}

	/**
	 * Return whether this is a training dataset or not.
	 * 
	 * @return
	 */
	public boolean isTraining() {
		return training;
	}

	/**
	 * @return the lenght of the longest sequence in this dataset.
	 */
	public int getMaxNumberOfTokens() {
		return maxNumberOfTokens;
	}

	/**
	 * Return the edge feature index for the given label. If a feature with such
	 * label does not exist, return <code>-1</code>.
	 * 
	 * @param label
	 * @return
	 */
	public int getEdgeFeatureIndex(String label) {
		for (int idx = 0; idx < featureLabelsEdge.length; ++idx)
			if (featureLabelsEdge[idx].equals(label))
				return idx;
		return -1;
	}

	/**
	 * Return the label for the given edge feature index.
	 * 
	 * @param index
	 * @return
	 */
	public String getEdgeFeatureLabel(int index) {
		return featureLabelsEdge[index];
	}

	/**
	 * Return the grandparent feature index for the given label. If a feature
	 * with such label does not exist, return <code>-1</code>.
	 * 
	 * @param label
	 * @return
	 */
	public int getGrandparentFeatureIndex(String label) {
		for (int idx = 0; idx < featureLabelsGrandparent.length; ++idx)
			if (featureLabelsGrandparent[idx].equals(label))
				return idx;
		return -1;
	}

	/**
	 * Return the label for the given grandparent feature index.
	 * 
	 * @param index
	 * @return
	 */
	public String getGrandparentFeatureLabel(int index) {
		return featureLabelsGrandparent[index];
	}

	/**
	 * Return the siblings feature index for the given label. If a feature with
	 * such label does not exist, return <code>-1</code>.
	 * 
	 * @param label
	 * @return
	 */
	public int getSiblingsFeatureIndex(String label) {
		for (int idx = 0; idx < featureLabelsSiblings.length; ++idx)
			if (featureLabelsSiblings[idx].equals(label))
				return idx;
		return -1;
	}

	/**
	 * Return the label for the given siblings feature index.
	 * 
	 * @param index
	 * @return
	 */
	public String getSiblingsFeatureLabel(int index) {
		return featureLabelsSiblings[index];
	}

	public void loadExamplesAndGenerate(String fileNameEdgeFactors,
			String fileNameGrandparentFactors,
			String fileNameLeftSiblingsFactors,
			String fileNameRightSiblingsFactors, DPGSModel model,
			long cacheSize, String fileNameSaveInputs) throws IOException,
			DatasetException, DPGSException {
		loadExamplesAndGenerate(fileNameEdgeFactors,
				fileNameGrandparentFactors, fileNameLeftSiblingsFactors,
				fileNameRightSiblingsFactors, null, model, cacheSize,
				fileNameSaveInputs);
	}

	public void loadExamplesAndGenerate(String fileNameEdgeFactors,
			String fileNameGrandparentFactors,
			String fileNameLeftSiblingsFactors,
			String fileNameRightSiblingsFactors, String[] templatesFilename,
			DPGSModel model, long cacheSize, String fileNameSaveInputs)
			throws IOException, DatasetException, DPGSException {

		BufferedReader readerEdge = new BufferedReader(new FileReader(
				fileNameEdgeFactors));
		BufferedReader readerGrandParent = new BufferedReader(new FileReader(
				fileNameGrandparentFactors));
		BufferedReader readerLeftSiblings = new BufferedReader(new FileReader(
				fileNameLeftSiblingsFactors));
		BufferedReader readerRightSiblings = new BufferedReader(new FileReader(
				fileNameRightSiblingsFactors));
		List<DPGSInput> listInput = new ArrayList<DPGSInput>(100);
		List<DPGSOutput> listOutput = new ArrayList<DPGSOutput>(100);

		// Load feature labels
		Set<Integer> multiValuedFeaturesIndexesEdge = loadFeatureLabelsEdge(readerEdge);
		Set<Integer> multiValuedFeaturesIndexesGrandparent = loadFeatureLabelsGrandparent(readerGrandParent);
		Set<Integer> multiValuedFeaturesIndexesLS = loadFeatureLabelsSiblings(readerLeftSiblings);
		Set<Integer> multiValuedFeaturesIndexesRS = loadFeatureLabelsSiblings(readerRightSiblings);

		// It was necessary to load template here, because to load template
		// is required to know the features labels.
		if (templatesFilename != null) {
			model.loadEdgeTemplates(templatesFilename[0], this);
			model.loadGrandparentTemplates(templatesFilename[1], this);
			model.loadLeftSiblingsTemplates(templatesFilename[2], this);
			model.loadRightSiblingsTemplates(templatesFilename[3], this);
		}
		boolean existExample = true;
		DPGSInput input;
		DPGSOutput output;
		int numberExample = 0;

		if (inputs == null) {
			inputs = new CacheExampleInputArray(cacheSize, fileNameSaveInputs);
		}

		System.out.println("Load examples");
		do {
			
			numberExample = listInput.size();
			
			existExample &= parseExample(readerEdge,
					multiValuedFeaturesIndexesEdge, listInput, listOutput,
					null, null);
			
			if(numberExample == listInput.size()){
				parseExample(readerGrandParent,
						multiValuedFeaturesIndexesGrandparent, listInput,
						listOutput, null, null);
				
				if (listInput.size() != numberExample) {
					throw new DatasetException(
							"The numbers of instances of grandparent file is different of edges file");
				}
				
				parseExample(readerLeftSiblings,
						multiValuedFeaturesIndexesGrandparent, listInput,
						listOutput, null, null);

				if (listInput.size() != numberExample) {
					throw new DatasetException(
							"The numbers of instances of left siblings file is different of edges file");
				}
				parseExample(readerRightSiblings,
						multiValuedFeaturesIndexesGrandparent, listInput,
						listOutput, null, null);

				if (listInput.size() != numberExample) {
					throw new DatasetException(
							"The numbers of instances of right siblings file is different of edges file");
				}

			} else {

				input = listInput.get(numberExample);
				output = listOutput.get(numberExample);

				parseExample(readerGrandParent,
						multiValuedFeaturesIndexesGrandparent, null, null,
						input, output);
				
				parseExample(readerLeftSiblings,
						multiValuedFeaturesIndexesLS, null, null, input, output);
				parseExample(readerRightSiblings,
						multiValuedFeaturesIndexesRS, null, null, input, output);

				model.generateFeaturesOneInput(input);
				input.cleanBasicFeatures();

				inputs.put(input);

				// Clean input of memory
				listInput.set(numberExample, null);
				input = null;

				numberExample++;

				if (numberExample % 100 == 0 && numberExample != 0) {
					System.out.print(".");
				}
			}
		} while (existExample);

		System.out.println("");

		outputs = listOutput.toArray(new DPGSOutput[0]);

		readerEdge.close();
		readerGrandParent.close();
		readerLeftSiblings.close();
		readerRightSiblings.close();
	}

	private Set<Integer> loadFeatureLabelsEdge(BufferedReader reader)
			throws IOException, DPGSException {
		// Parse feature labels in the first line of the file.
		String[] tmpFeatureLabelsEdge = parseFeatureLabels(reader.readLine());

		// Skip blank line after label header.
		reader.readLine();

		if (featureLabelsEdge == null) {
			// This is the first grandparent dataset that has been loaded.
			featureLabelsEdge = tmpFeatureLabelsEdge;
		} else {
			// Check if the previous datasets have exactly the same feature set.
			if (!Arrays.equals(tmpFeatureLabelsEdge, featureLabelsEdge))
				throw new DPGSException("Given edge dataset has a "
						+ "different feature set from previous one(s)");
		}

		// Multi-valued features indexes.
		Set<Integer> multiValuedFeaturesIndexes = new TreeSet<Integer>();
		for (String label : multiValuedEdgeFeatures)
			multiValuedFeaturesIndexes.add(getEdgeFeatureIndex(label));
		return multiValuedFeaturesIndexes;
	}

	private Set<Integer> loadFeatureLabelsGrandparent(BufferedReader reader)
			throws IOException, DPGSException {
		// Parse feature labels in the first line of the file.
		String[] tmpFeatureLabelsGrandparent = parseFeatureLabels(reader
				.readLine());

		// Skip blank line after label header.
		reader.readLine();

		if (featureLabelsGrandparent == null) {
			// This is the first grandparent dataset that has been loaded.
			featureLabelsGrandparent = tmpFeatureLabelsGrandparent;
		} else {
			// Check if the previous datasets have exactly the same feature set.
			if (!Arrays.equals(tmpFeatureLabelsGrandparent,
					featureLabelsGrandparent))
				throw new DPGSException("Given grandparent dataset has a "
						+ "different feature set from previous one(s)");
		}

		// Multi-valued features indexes.
		Set<Integer> multiValuedFeaturesIndexes = new TreeSet<Integer>();
		for (String label : multiValuedGrandparentFeatures)
			multiValuedFeaturesIndexes.add(getGrandparentFeatureIndex(label));
		return multiValuedFeaturesIndexes;
	}

	private Set<Integer> loadFeatureLabelsSiblings(BufferedReader reader)
			throws IOException, DPGSException {
		// Parse feature labels in the first line of the file.
		String[] tmpFeatureLabelsSiblings = parseFeatureLabels(reader
				.readLine());

		// Skip blank line after label header.
		reader.readLine();

		if (featureLabelsSiblings == null) {
			// This is the first grandparent dataset that has been loaded.
			featureLabelsSiblings = tmpFeatureLabelsSiblings;
		} else {
			// Check if the previous datasets have exactly the same feature set.
			if (!Arrays.equals(tmpFeatureLabelsSiblings, featureLabelsSiblings))
				throw new DPGSException("Given siblings dataset has a "
						+ "different feature set from previous one(s)");
		}

		// Multi-valued features indexes.
		Set<Integer> multiValuedFeaturesIndexes = new TreeSet<Integer>();
		for (String label : multiValuedSiblingsFeatures)
			multiValuedFeaturesIndexes.add(getSiblingsFeatureIndex(label));
		return multiValuedFeaturesIndexes;
	}

	/**
	 * Parse line with feature labels that must be of the form:
	 * 
	 * [features = label1, label2, ..., labelM]
	 * 
	 * @param line
	 * @return
	 */
	protected String[] parseFeatureLabels(String line) {
		int eq = line.indexOf('=');
		int end = line.indexOf(']');
		String[] labels = line.substring(eq + 1, end).split(",");
		String[] featureLabels = new String[labels.length - 2];
		for (int i = 1; i < labels.length - 1; ++i)
			featureLabels[i - 1] = labels[i].trim();
		return featureLabels;
	}

	/**
	 * Save dataset to the given file.
	 * 
	 * @param fileName
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void save(String fileName) throws IOException, DatasetException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		save(writer);
		writer.close();
	}

	/**
	 * Save dataset to the given output stream.
	 * 
	 * @param os
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void save(OutputStream os) throws IOException, DatasetException {
		save(new BufferedWriter(new OutputStreamWriter(os)));
	}

	/**
	 * Save dataset to the given buffered writer.
	 * 
	 * @param writer
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void save(BufferedWriter writer) throws IOException,
			DatasetException {
		throw new NotImplementedException();
	}

	/**
	 * Save this dataset along with a new column with the given predicted
	 * outputs.
	 * 
	 * @param inputFilename
	 * @param outputFilename
	 * @param predictedOuputs
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void save(String inputFilename, String outputFilename,
			DPGSOutput[] predictedOuputs) throws IOException, DatasetException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(
				outputFilename));
		BufferedReader reader = new BufferedReader(
				new FileReader(inputFilename));
		save(reader, writer, predictedOuputs);
		reader.close();
		writer.close();
	}

	/**
	 * Save this dataset along with a new column with the given predicted
	 * outputs.
	 * 
	 * @param reader
	 *            file with the CoNLL-format data where to append the predicted
	 *            output column.
	 * @param writer
	 *            output writer to save the predicted outputs along with the
	 *            input columns.
	 * @param predictedOuputs
	 *            array of predicted outputs.
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void save(BufferedReader reader, BufferedWriter writer,
			DPGSOutput[] predictedOuputs) throws IOException, DatasetException {
		// Number of given output structures.
		int numExs = predictedOuputs.length;
		// Index of the current example.
		int idxEx = 0;
		while (true) {
			String lastLine = null;
			DPGSOutput output = predictedOuputs[idxEx];
			int numTokens = output.size();
			for (int idxMod = 1; idxMod < numTokens; ++idxMod) {
				String line = lastLine = reader.readLine();
				if (line == null)
					throw new DatasetException("Missing lines in input file");
				if (line.trim().length() == 0)
					throw new DatasetException(String.format(
							"Example %d has length %d in the dataset "
									+ "file but has length %d in the "
									+ "corpus file", idxEx, numTokens, idxMod));
				String[] ftrs = line.split("[ ]");
				int head = output.getHead(idxMod);
				if (head < 0)
					head = 0;
				// Write original input features (skip two last output columns).
				for (int idxFtr = 0; idxFtr < ftrs.length - 2; ++idxFtr)
					writer.write(String.format("%s ", ftrs[idxFtr]));
				// Write response columns (dependency label is always PUNC).
				writer.write(String.format("%d PUNC\n", head));
			}

			++idxEx;

			// Check boundary line.
			String line = reader.readLine();
			if (line == null) {
				if (idxEx != numExs)
					throw new DatasetException("Missing lines in input file");
				else
					break;
			} else {
				if (line.trim().length() > 0) {
					throw new DatasetException("More lines than expected in "
							+ "input file");
				} else if (idxEx == numExs) {
					/*
					 * Make sure output file has the same length of input. This
					 * is a requirement of the official CoNLL evaluation script.
					 */
					writer.write("\n");

					// Make sure all remaining lines are blank.
					while ((line = reader.readLine()) != null) {
						if (line.trim().length() > 0) {
							throw new DatasetException(
									"More lines than expected in "
											+ "input file");
						}
						/*
						 * Make sure output file has the same length of input.
						 * This is a requirement of the official CoNLL
						 * evaluation script.
						 */
						writer.write("\n");
					}
					break;
				}
			}

			writer.write("\n");
		}
	}

	/**
	 * Parse an example from the given reader.
	 * 
	 * An example is a sequence of factors (edge, siblings or grandparent
	 * factors). Each factor is represented in one line. A blank line separates
	 * one example from the next one. Each factor (non-blank line) is a sequence
	 * of feature values (in the order that was presented in the file header).
	 * There is an additional factor type (LEN) which is used to indicate the
	 * sentence length.
	 * 
	 * The first feature of each factor comprises its ID that *must* obey the
	 * following format: "TYPE(IDX1,IDX2,IDX3)". TYPE can be E for edge factors,
	 * G for grandparent, S for siblings or LEN for the special factor that
	 * indicates the sentence length. For all factor types but LEN, IDX1 and
	 * IDX2 indicate, respectivelly, the head token index and the modifier token
	 * index. For grandparent factors, IDX3 is the grandparent token index. For
	 * siblings factors, IDX3 is the previous modifier (sibling) token index.
	 * For edge factors, IDX3 is irrelevant. For LEN factors, IDX1, IDX2 and
	 * IDX3 are all the same: the sentence length.
	 * 
	 * The last feature of a factor must be "Y" if the factor is part of the
	 * correct dependecy tree or "N" otherwise. The remaining values are the
	 * ordinary basic features.
	 * 
	 * Ommited factors will constrain the feasible outputs for the example. That
	 * is, no feasible solution can include an inexistent factor.
	 * 
	 * @param reader
	 *            input file reader positioned at the beginning of an example or
	 *            at the end of the file.
	 * @param multiValuedFeatureIndexes
	 *            which features are multi-valued features.
	 * @param valueSeparator
	 *            character sequence that separates values within a multi-valued
	 *            feature.
	 * @param inputList
	 *            the list of input structures to store the read input.
	 * @param outputList
	 *            the list of output structures to store the read output.
	 * @param input
	 *            if inputList and outputList are null, this value will give the
	 *            input structure corresponding to the current example in the
	 *            given reader.
	 * @param output
	 *            if inputList and outputList are null, this value will give the
	 *            output structure corresponding to the current example in the
	 *            given reader.
	 * @return
	 * @throws IOException
	 *             if there is a problem reading the input file.
	 * @throws DatasetException
	 *             if there is a syntax or semantic issue.
	 * @throws DPGSException
	 */
	protected boolean parseExample(BufferedReader reader,
			Set<Integer> multiValuedFeatureIndexes, List<DPGSInput> inputList,
			List<DPGSOutput> outputList, DPGSInput input, DPGSOutput output)
			throws IOException, DatasetException, DPGSException {
		/*
		 * List of factors for one example. Each factor is a list of lists of
		 * integer arrays. Feature values can be multi-valued. Thus, each
		 * feature value is represented by an integer array. A factor is then
		 * represented by a list of integer arrays. Finally, the following
		 * variable stores a list of such factors.
		 */
		LinkedList<LinkedList<int[]>> factors = new LinkedList<LinkedList<int[]>>();

		// Parameters of correct factors.
		LinkedList<int[]> correctFactors = new LinkedList<int[]>();

		// Number of tokens in the sentence.
		int numTokens = -1;

		// Maximum token index.
		int maxTokenIndex = -1;

		// Number of features.
		int numFeatures = -1;

		// Read lines up to a blank line.
		String line;
		while ((line = reader.readLine()) != null) {

			line = line.trim();
			if (line.length() == 0)
				// Stop on blank lines.
				break;

			// Split factor in feature values.
			String[] ftrValues = REGEX_SPACE.split(line);

			// Check (or set) number of features.
			if (numFeatures == -1)
				numFeatures = ftrValues.length;
			else if (numFeatures != ftrValues.length)
				throw new DatasetException(
						String.format(
								"Number of features in example %d is equal to %d but should be %d",
								inputList.size(), ftrValues.length, numFeatures));

			int[] idFactor = new int[4];
			idFactor[0] = -1;

			// Factor ID: type and parameters (index).
			Matcher m = REGEX_ID.matcher(ftrValues[0]);
			if (!m.matches())
				throw new DatasetException(
						String.format(
								"ID (%s) of the factor does not match "
										+ "expected format which is TYPE(IDX1,IDX2,IDX3)",
								ftrValues[0]));

			/*
			 * First parameter is the factor type: E (edge), G (grandparent), S
			 * (siblings) or LEN (sentence length).
			 */
			String type = m.group(1);
			if ("E".equals(type))
				idFactor[0] = 0;
			else if ("G".equals(type))
				idFactor[0] = 1;
			else if ("S".equals(type))
				idFactor[0] = 2;
			else if ("LEN".equals(type))
				// Special factor just to indicate the length of the sentence.
				idFactor[0] = -1;
			else
				throw new DatasetException(String.format(
						"Factor type should be E, G, S or LEN, but it is %s",
						type));

			// Factor remaining parameters.
			idFactor[1] = Integer.parseInt(m.group(2));
			idFactor[2] = Integer.parseInt(m.group(3));
			idFactor[3] = Integer.parseInt(m.group(4));

			if (idFactor[0] == -1) {
				/*
				 * For LEN factors, all parameters are equal to the sentence
				 * length.
				 */
				if (idFactor[1] - 1 > maxTokenIndex)
					maxTokenIndex = idFactor[1] - 1;
				// LEN factor is useful only to indicate the sentence length.
				continue;
			} else if (idFactor[0] == 0) {
				// For E factors, the last parameter is the sentence length.
				if (idFactor[3] - 1 > maxTokenIndex)
					maxTokenIndex = idFactor[3] - 1;
			} else if (idFactor[0] == 1) {
				// For G factors, all parameters are ordinary token indexes.
				if (idFactor[1] > maxTokenIndex)
					maxTokenIndex = idFactor[1];
				if (idFactor[2] > maxTokenIndex)
					maxTokenIndex = idFactor[2];
				if (idFactor[3] > maxTokenIndex)
					maxTokenIndex = idFactor[3];
			} else if (idFactor[0] == 2) {
				/*
				 * For S factors, the first parameter is an ordinary token index
				 * but the remaining parameters can be equal to the sentence
				 * lenght.
				 */
				if (idFactor[1] > maxTokenIndex)
					maxTokenIndex = idFactor[1];
				if (idFactor[2] - 1 > maxTokenIndex)
					maxTokenIndex = idFactor[2] - 1;
				if (idFactor[3] - 1 > maxTokenIndex)
					maxTokenIndex = idFactor[3] - 1;
			}

			// Factor features.
			LinkedList<int[]> factor = new LinkedList<int[]>();

			// The first feature comprises the factor parameters.
			factor.add(idFactor);

			// Encode the factor basic features.
			for (int idxFtr = 1; idxFtr < numFeatures - 1; ++idxFtr) {
				int[] vals;
				String str = ftrValues[idxFtr];
				if (multiValuedFeatureIndexes.contains(idxFtr)) {
					// Split multi-valued feature.
					String[] valsStr = str.split(separatorFeatureValues);
					vals = new int[valsStr.length];
					for (int idxVal = 0; idxVal < vals.length; ++idxVal)
						vals[idxVal] = basicEncoding.put(new String(
								valsStr[idxVal]));
				} else {
					// Single-valued feature.
					vals = new int[1];
					vals[0] = basicEncoding.put(new String(str));
				}
				factor.add(vals);
			}

			// The last value is the correct factor flag: Y or N.
			String isCorrectEdge = ftrValues[numFeatures - 1];
			if (isCorrectEdge.equals("Y")) {
				// Add factor parameters to the list of correct factors.
				correctFactors.add(idFactor);
			} else if (!isCorrectEdge.equals("N")) {
				// Flag is neither Y nor N.
				throw new DatasetException(String.format(
						"Last feature value must be Y or N to indicate "
								+ "the correct edge. However, for factor "
								+ " %s(%d,%d,%d) this feature value is %s ",
						type, idFactor[1], idFactor[2], idFactor[3],
						isCorrectEdge));
			}

			// Add built factor.
			factors.add(factor);
		}

		if (maxTokenIndex == -1)
			// Extra empty line or end of file (when line == null).
			return line != null;

		if (numTokens == -1)
			numTokens = maxTokenIndex + 1;

		if (input == null) {
			// Id is just the example index.
			String id = "" + inputList.size();
			// Create new input structure, if it is not given.
			input = new DPGSInput(id, numTokens);
		} else if (numTokens != input.size())
			throw new DPGSException(String.format(
					"Incorrect number of token in example %s", input.getId()));

		// Fill the basic features of the given factors.
		input.addBasicFeaturesOfFactors(factors);

		// Keep the length of the longest example.
		if (numTokens > maxNumberOfTokens)
			maxNumberOfTokens = numTokens;

		if (output == null) {
			// Create a new output structure, if not given.
			output = input.createOutput();
			Arrays.fill(output.getHeads(), -1);
			Arrays.fill(output.getGrandparents(), -1);
		}

		// Fill and check the output structure.
		for (int[] params : correctFactors) {
			int type = params[0];
			int idxHead = params[1];
			int idxMod = params[2];
			if (type == 0) {
				// EDGE factor.
				setAndCheckEdgeVariable(output, input.getId(), params, idxHead,
						idxMod);
			} else if (type == 1) {
				// GRANDPARENT factor.
				int idxGrandparent = params[3];
				if (idxGrandparent == idxHead || idxMod == idxHead)
					continue;
				// Parse tree variable.
				setAndCheckEdgeVariable(output, input.getId(), params, idxHead,
						idxMod);
				// Grandparent and modifier variables.
				setAndCheckEdgeVariable(output, input.getId(), params,
						idxGrandparent, idxHead);
			} else if (type == 2) {
				// SIBLINGS factor.
				int idxPrevMod = params[3];
				if (idxMod != idxHead && idxMod != numTokens) {
					// Set modifier.
					output.setModifier(idxHead, idxMod, true);
					// Check dependency (idxHead, idxMod).
					setAndCheckEdgeVariable(output, input.getId(), params,
							idxHead, idxMod);
				}
				if (idxPrevMod != idxHead && idxPrevMod != numTokens) {
					// Set previous modifier.
					output.setModifier(idxHead, idxPrevMod, true);
					// Check dependency (idxHead, idxMod).
					setAndCheckEdgeVariable(output, input.getId(), params,
							idxHead, idxPrevMod);
				}
				output.setPreviousModifier(idxHead, idxMod, idxPrevMod);
			}
		}

		if (inputList != null) {
			inputList.add(input);
			outputList.add(output);
		}

		// Return true if there are more lines.
		return line != null;
	}

	/**
	 * Set grandparent and head properties and check their consistency for the
	 * given dependency.
	 * 
	 * @param output
	 * @param id
	 * @param params
	 * @param idxHead
	 * @param idxModifier
	 * @throws DatasetException
	 */
	protected void setAndCheckEdgeVariable(DPGSOutput output, String id,
			int[] params, int idxHead, int idxModifier) throws DatasetException {
		// Set and check head property.
		if (output.getHead(idxModifier) == -1)
			output.setHead(idxModifier, idxHead);
		else if (output.getHead(idxModifier) != idxHead)
			throw new DatasetException(String.format(
					"Incosistent parameter %d(%d,%d,%d) for example %s",
					params[0], params[1], params[2], params[3], id));
		// Set and check grandparent property.
		if (output.getGrandparent(idxModifier) == -1)
			output.setGrandparent(idxModifier, idxHead);
		else if (output.getGrandparent(idxModifier) != idxHead)
			throw new DatasetException(String.format(
					"Incosistent parameter %d(%d,%d,%d) for example %s",
					params[0], params[1], params[2], params[3], id));
	}

	/**
	 * Set the modifier variables for all output structures in this dataset. It
	 * is assumed that all variables are equal to <code>false</code>.
	 */
	public void setModifierVariables() {
		int numExs = outputs.length;
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			setModifierVariablesOneOutput(idxEx);
		}
	}

	private void setModifierVariablesOneOutput(int idxEx) {
		DPGSOutput output = outputs[idxEx];
		int numTkns = output.size();
		for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
			int idxHead = output.getHead(idxModifier);
			if (idxHead != idxModifier && idxHead != -1)
				output.setModifier(idxHead, idxModifier, true);
		}
	}

	/**
	 * Skip blank lines and lines starting by the comment character #.
	 * 
	 * @param reader
	 * @return
	 * @throws IOException
	 */
	public static String skipBlanksAndComments(BufferedReader reader)
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
	 * Return the basic feature encoding.
	 * 
	 * @return
	 */
	public FeatureEncoding<String> getFeatureEncoding() {
		return basicEncoding;
	}

	public void saveCore(ObjectOutputStream stream) throws IOException {

		stream.writeObject(multiValuedEdgeFeatures
				.toArray(new String[multiValuedEdgeFeatures.size()]));
		stream.writeObject(multiValuedGrandparentFeatures
				.toArray(new String[multiValuedGrandparentFeatures.size()]));
		stream.writeObject(multiValuedSiblingsFeatures
				.toArray(new String[multiValuedSiblingsFeatures.size()]));
		stream.writeObject(separatorFeatureValues);

		stream.writeObject(basicEncoding);
	}

	static public DPGSDataset loadCore(ObjectInputStream stream)
			throws IOException, ClassNotFoundException {

		String[] multiValuedEdgeFeatures = (String[]) stream.readObject();
		String[] multiValuedGrandparentFeatures = (String[]) stream
				.readObject();
		String[] multiValuedSiblingsFeatures = (String[]) stream.readObject();
		String separatorFeatureValues = (String) stream.readObject();
		FeatureEncoding<String> basicEncoding = (FeatureEncoding<String>) stream
				.readObject();

		return new DPGSDataset(multiValuedEdgeFeatures,
				multiValuedGrandparentFeatures, multiValuedSiblingsFeatures,
				separatorFeatureValues, basicEncoding);
	}

	@Override
	public void load(String fileName) throws IOException, DatasetException {
		throw new NotImplementedException();
	}

	@Override
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		throw new NotImplementedException();
	}

	@Override
	public void load(InputStream is) throws IOException, DatasetException {
		throw new NotImplementedException();
	}

	private void loadBasicEncoding() throws IOException, ClassNotFoundException {
		File file = new File("basicEncodings");
		if (!file.exists()) {
			if (basicEncoding == null)
				basicEncoding = new StringMapEncoding();
		}

		FileInputStream fileIn = null;
		BufferedInputStream buf = null;
		ObjectInputStream objInput = null;

		try {
			fileIn = new FileInputStream(file);
			buf = new BufferedInputStream(fileIn);
			objInput = new ObjectInputStream(buf);

			basicEncoding = (FeatureEncoding<String>) objInput.readObject();
		} finally {
			if (objInput != null)
				objInput.close();
			else if (buf != null)
				buf.close();
			else if (fileIn != null)
				fileIn.close();
		}
	}

	private void unloadBasicEncoding() throws IOException,
			ClassNotFoundException {

		FileOutputStream fileOut = null;
		BufferedOutputStream bufOut = null;
		ObjectOutputStream objOut = null;

		try {
			fileOut = new FileOutputStream("basicEncodings");
			bufOut = new BufferedOutputStream(fileOut);
			objOut = new ObjectOutputStream(fileOut);

			objOut.writeObject(basicEncoding);
		} finally {
			if (objOut != null)
				objOut.close();
			else if (bufOut != null)
				bufOut.close();
			else if (fileOut != null)
				fileOut.close();
		}

		basicEncoding = null;
	}
}
