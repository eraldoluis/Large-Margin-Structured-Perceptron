package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
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
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
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
public class DPGSDataset {

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
			.compile("^(G|S|LEN)\\((\\d+),(\\d+),(\\d+)\\)$");

	/**
	 * Encoding for basic textual features (column-format features).
	 */
	protected FeatureEncoding<String> basicEncoding;

	/**
	 * Basic feature labels for grandparent factors.
	 */
	protected String[] featureLabelsGrandparent;

	/**
	 * Basic feature labels for siblings factors.
	 */
	protected String[] featureLabelsSiblings;

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
	protected DPGSInput[] inputs;

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
	public DPGSDataset(String[] multiValuedGrandparentFeatures,
			String[] multiValuedSiblingsFeatures, String separatorFeatureValues) {
		this.basicEncoding = new StringMapEncoding();
		this.multiValuedGrandparentFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedGrandparentFeatures.length; ++idxFtr)
			this.multiValuedGrandparentFeatures
					.add(multiValuedGrandparentFeatures[idxFtr]);
		this.multiValuedSiblingsFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedSiblingsFeatures.length; ++idxFtr)
			this.multiValuedSiblingsFeatures
					.add(multiValuedSiblingsFeatures[idxFtr]);
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
	public DPGSDataset(String[] multiValuedGrandparentFeatures,
			String[] multiValuedSiblingsFeatures,
			String separatorFeatureValues, FeatureEncoding<String> basicEncoding) {
		this.basicEncoding = basicEncoding;
		this.multiValuedGrandparentFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedGrandparentFeatures.length; ++idxFtr)
			this.multiValuedGrandparentFeatures
					.add(multiValuedGrandparentFeatures[idxFtr]);
		this.multiValuedSiblingsFeatures = new TreeSet<String>();
		for (int idxFtr = 0; idxFtr < multiValuedSiblingsFeatures.length; ++idxFtr)
			this.multiValuedSiblingsFeatures
					.add(multiValuedSiblingsFeatures[idxFtr]);
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

	/**
	 * Return array of input structures (sentence factors).
	 * 
	 * @return
	 */
	public DPGSInput[] getInputs() {
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
		return inputs[index];
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
		return inputs.length;
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
	 * Load dataset from the given file.
	 * 
	 * @param fileName
	 * @throws IOException
	 * @throws DatasetException
	 * @throws DPGSException
	 */
	public void loadGrandparentFactors(String fileName) throws IOException,
			DatasetException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		loadGrandparentFactors(reader);
		reader.close();
	}

	/**
	 * Load dataset from the given buffered reader.
	 * 
	 * @param reader
	 * @throws IOException
	 * @throws DatasetException
	 * @throws DPGSException
	 */
	public void loadGrandparentFactors(BufferedReader reader)
			throws IOException, DatasetException, DPGSException {
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

		// Load example factors from the given reader.
		loadExampleFactors(reader, multiValuedFeaturesIndexes);
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

	/**
	 * Load siblings dataset from the given file.
	 * 
	 * @param fileName
	 * @throws IOException
	 * @throws DatasetException
	 * @throws DPGSException
	 */
	public void loadSiblingsFactors(String fileName) throws IOException,
			DatasetException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		loadSiblingsFactors(reader);
		reader.close();
	}

	/**
	 * Load siblings dataset from the given buffered reader.
	 * 
	 * @param reader
	 * @throws IOException
	 * @throws DatasetException
	 * @throws DPGSException
	 */
	public void loadSiblingsFactors(BufferedReader reader) throws IOException,
			DatasetException, DPGSException {
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

		// Load example factors from the given reader.
		loadExampleFactors(reader, multiValuedFeaturesIndexes);
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
	 * Load example factors from the given reader.
	 * 
	 * @param reader
	 * @param multiValuedFeaturesIndexes
	 * @throws IOException
	 * @throws DatasetException
	 * @throws DPGSException
	 */
	protected void loadExampleFactors(BufferedReader reader,
			Set<Integer> multiValuedFeaturesIndexes) throws IOException,
			DatasetException, DPGSException {
		List<DPGSInput> inputList = null;
		List<DPGSOutput> outputList = null;
		if (inputs == null) {
			inputList = new LinkedList<DPGSInput>();
			outputList = new LinkedList<DPGSOutput>();
		}

		int numExs = 0;
		DPGSInput input = null;
		DPGSOutput output = null;
		if (inputs != null) {
			input = inputs[0];
			output = outputs[0];
		}

		// Parse examples and create new inputs/outputs or fill existing ones.
		while (parseExample(reader, multiValuedFeaturesIndexes, inputList,
				outputList, input, output)) {
			++numExs;
			if ((numExs + 1) % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}

			if (inputs != null && numExs < inputs.length) {
				input = inputs[numExs];
				output = outputs[numExs];
			} else {
				input = null;
				output = null;
			}
		}
		System.out.println();

		// Convert list of structures to arrays.
		if (inputs == null) {
			inputs = inputList.toArray(new DPGSInput[0]);
			outputs = outputList.toArray(new DPGSOutput[0]);
		}

		LOG.info("Read " + inputs.length + " examples.");
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
			DPGSOutput output = predictedOuputs[idxEx];
			int numTokens = output.size();
			for (int idxMod = 1; idxMod < numTokens; ++idxMod) {
				String line = reader.readLine();
				if (line == null)
					throw new DatasetException("Missing lines in input file");
				if (line.trim().length() == 0)
					throw new DatasetException("Example is shorter than "
							+ "expected in input file");
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
				if (line.trim().length() > 0)
					throw new DatasetException("More lines than expected in "
							+ "input file");
				else if (idxEx == numExs) {
					/*
					 * Make sure output file has the same length of input. This
					 * is a requirement of the official CoNLL evaluation script.
					 */
					writer.write("\n");

					// Make sure all remaining lines are blank.
					while ((line = reader.readLine()) != null) {
						if (line.trim().length() > 0)
							throw new DatasetException(
									"More lines than expected in "
											+ "input file");
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
	 * An example is a sequence of edges. Each edge is represented in one line.
	 * Blank lines separate one example from the next one. Each line (edge) is a
	 * sequence of feature values (in the order that was presented in the file
	 * header).
	 * 
	 * The first feature of each edge comprises its ID that *must* obey the
	 * format "[head token index]>[dependent token index]" to indicate end
	 * points of the directed edge. The last feature is equal to "TRUE" if the
	 * edge is part of the correct dependecy tree of this example and "FALSE"
	 * otherwise. The remaining values are the ordinary basic features.
	 * 
	 * Edge can be ommited and then will be considered inexistent.
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
		 * List of factors. Each line of the input file contains a factor. Each
		 * factor comprises a fixed sequence of features. The fisrt feature is a
		 * identification comprising the type of the factor (G for grandparent
		 * or S for siblings) followed by three integer parameters between
		 * parenthesis. The parameters are (idxHead,idxModifier,idxGrandparent
		 * or idxPrevModifier). The following, and ordinary, features are
		 * textual representations that can even contains more than one value.
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

			// Type and parameters.
			Matcher m = REGEX_ID.matcher(ftrValues[0]);
			if (!m.matches())
				throw new DatasetException(
						"ID of the factor does not match expected format");
			int[] params = new int[4];

			// Factor type: G or S.
			String type = m.group(1);
			if (type.equals("G"))
				params[0] = 1;
			else if (type.equals("S"))
				params[0] = 2;
			else if (type.equals("LEN"))
				// Special factor just to indicate the length of the sentence.
				params[0] = -1;
			else
				throw new DatasetException(String.format(
						"Factor type (%s) should be G or S.", type));
			// Factor remaining parameters.
			params[1] = Integer.parseInt(m.group(2));
			params[2] = Integer.parseInt(m.group(3));
			params[3] = Integer.parseInt(m.group(4));

			// Update maximum token index.
			if (params[1] > maxTokenIndex)
				maxTokenIndex = params[1];
			/*
			 * Ignore second and third parameter of sibling factors (it can be
			 * equal to lenSent).
			 */
			if (params[0] != 2) {
				if (params[2] > maxTokenIndex)
					maxTokenIndex = params[2];
				if (params[3] > maxTokenIndex)
					maxTokenIndex = params[3];
			}

			if (params[0] == -1)
				// Special factor just to indicate the length of this sentence.
				continue;

			// Factor features.
			LinkedList<int[]> factor = new LinkedList<int[]>();

			// The first value comprises the factor params.
			factor.add(params);

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
				correctFactors.add(params);
			} else if (!isCorrectEdge.equals("N")) {
				// Flag is neither Y nor N.
				throw new DatasetException(String.format(
						"Last feature value must be Y or N to indicate "
								+ "the correct edge. However, for factor "
								+ " %s(%d,%d,%d) this feature value is %s ",
						type, params[1], params[2], params[3], isCorrectEdge));
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
		input.fillBasicFeatures(factors);

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
			int idxHead = params[1];
			int idxMod = params[2];
			if (params[0] == 1) {
				// Grandparent factor.
				int idxGrandparent = params[3];
				setAndCheckDependency(output, input.getId(), params, idxHead,
						idxMod);
				setAndCheckDependency(output, input.getId(), params,
						idxGrandparent, idxHead);
			} else {
				// Sibling factor.
				int idxPrevMod = params[3];
				if (idxMod != idxHead && idxMod != numTokens) {
					// Set modifier.
					output.setModifier(idxHead, idxMod, true);
					// Check dependency (idxHead, idxMod).
					setAndCheckDependency(output, input.getId(), params,
							idxHead, idxMod);
				}
				if (idxPrevMod != idxHead && idxPrevMod != numTokens) {
					// Set previous modifier.
					output.setModifier(idxHead, idxPrevMod, true);
					// Check dependency (idxHead, idxMod).
					setAndCheckDependency(output, input.getId(), params,
							idxHead, idxPrevMod);
				}
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
	protected void setAndCheckDependency(DPGSOutput output, String id,
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
}
