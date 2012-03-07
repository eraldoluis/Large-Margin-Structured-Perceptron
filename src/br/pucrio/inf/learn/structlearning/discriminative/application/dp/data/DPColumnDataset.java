package br.pucrio.inf.learn.structlearning.discriminative.application.dp.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.FeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.InvertedIndex;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.SimpleFeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a dataset with dependency parsing examples. Each edge within a
 * sentence is represented by a fixed number of features (column-based
 * representation, also commonly used in CoNLL shared tasks). This class is
 * useful for template-based models.
 * 
 * In this representation, there are two types of features: basic and explicit.
 * Basic features come from dataset columns. Each column has a name and an
 * sequential id, and, for each example, there is a string value for each basic
 * feature (column). Explicit features are instantiated from templates. These
 * are the real features, that is the features used in the model. Generally,
 * each template combines some basic features and generates an explicit feature
 * for each edge.
 * 
 * There are also two encodings. The first one encodes basic features textual
 * values into integer codes. The second one encodes explicit features (i.e.,
 * combined basic features by means of templates) values into integer codes.
 * These later codes are used by the model as parameter indexes, i.e., for each
 * index, the model includes a learned weight.
 * 
 * @author eraldo
 * 
 */
public class DPColumnDataset implements DPDataset {

	/**
	 * Logging object.
	 */
	private static Log LOG = LogFactory.getLog(DPColumnDataset.class);

	/**
	 * Regular expression pattern to parse spaces.
	 */
	private final Pattern REGEX_SPACE = Pattern.compile("[ ]");

	/**
	 * Encoding for basic textual features.
	 */
	private FeatureEncoding<String> basicEncoding;

	/**
	 * Template set partitions.
	 */
	private FeatureTemplate[][] templates;

	/**
	 * Encoding for explicit features that are created from templates by
	 * conjoining basic features.
	 */
	private MapEncoding<Feature> explicitEncoding;

	/**
	 * Number of template partitions.
	 */
	private int numberOfPartitions;

	/**
	 * Current partition.
	 */
	private int currentPartition;

	/**
	 * Feature labels.
	 */
	private String[] featureLabels;

	/**
	 * Multi-valued features.
	 */
	private Set<String> multiValuedFeatures;

	/**
	 * Input sequences.
	 */
	private DPInput[] inputs;

	/**
	 * Output branchings.
	 */
	private DPOutput[] outputs;

	/**
	 * Indicate if this dataset is a training set.
	 */
	private boolean training;

	/**
	 * Length of the longest sequence in this dataset.
	 */
	private int maxNumberOfTokens;

	/**
	 * Optional inverted index representation to speedup some algorithms.
	 * Mainly, training algorithms.
	 */
	private InvertedIndex invertedIndex;

	/**
	 * Punctuation file reader.
	 */
	private BufferedReader readerPunc;

	/**
	 * Punctuation file name.
	 */
	private String fileNamePunc;

	/**
	 * Create edge corpus.
	 * 
	 * @param multiValuedFeatures
	 */
	public DPColumnDataset(Collection<String> multiValuedFeatures) {
		this.multiValuedFeatures = new TreeSet<String>();
		if (multiValuedFeatures != null) {
			for (String ftrLabel : multiValuedFeatures)
				this.multiValuedFeatures.add(ftrLabel);
		}
		this.basicEncoding = new StringMapEncoding();
		this.explicitEncoding = new MapEncoding<Feature>();
	}

	/**
	 * Create a new dataset using the same encoding and other underlying data
	 * structures of the given 'sibling' dataset.
	 * 
	 * For most use cases, the underlying data structures within the sibling
	 * dataset should be kept unchanged after creating this new dataset.
	 * 
	 * @param sibling
	 */
	public DPColumnDataset(DPColumnDataset sibling) {
		this.multiValuedFeatures = sibling.multiValuedFeatures;
		this.basicEncoding = sibling.basicEncoding;
		this.explicitEncoding = sibling.explicitEncoding;
		this.templates = sibling.templates;
	}

	/**
	 * @return the number of features (columns) in this corpus.
	 */
	public int getNumberOfFeatures() {
		return featureLabels.length;
	}

	@Override
	public DPInput[] getInputs() {
		return inputs;
	}

	@Override
	public DPOutput[] getOutputs() {
		return outputs;
	}

	@Override
	public DPInput getInput(int index) {
		return inputs[index];
	}

	@Override
	public DPOutput getOutput(int index) {
		return outputs[index];
	}

	@Override
	public int getNumberOfExamples() {
		return inputs.length;
	}

	@Override
	public boolean isTraining() {
		return training;
	}

	/**
	 * Return the optional inverted index that represents this corpus.
	 * 
	 * This data structure can be used to speedup training algorithms.
	 * 
	 * @return
	 */
	public InvertedIndex getInvertedIndex() {
		return invertedIndex;
	}

	/**
	 * Create an inverted index for this corpus.
	 */
	public void createInvertedIndex() {
		this.invertedIndex = new InvertedIndex(this);
	}

	/**
	 * Return the punctuation file name.
	 * 
	 * @return
	 */
	public String getFileNamePunc() {
		return fileNamePunc;
	}

	/**
	 * Set the name of the punctuation file for this edge corpus.
	 * 
	 * @param filename
	 */
	public void setFileNamePunc(String filename) {
		this.fileNamePunc = filename;
	}

	/**
	 * @return the lenght of the longest sequence in this dataset.
	 */
	public int getMaxNumberOfTokens() {
		return maxNumberOfTokens;
	}

	@Override
	public void load(String fileName) throws IOException, DatasetException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		load(reader);
		reader.close();
	}

	@Override
	public void load(InputStream is) throws IOException, DatasetException {
		load(new BufferedReader(new InputStreamReader(is)));
	}

	/**
	 * Return the feature index for the given label. If a feature with such
	 * label does not exist, return <code>-1</code>.
	 * 
	 * @param label
	 * @return
	 */
	public int getFeatureIndex(String label) {
		for (int idx = 0; idx < featureLabels.length; ++idx)
			if (featureLabels[idx].equals(label))
				return idx;
		return -1;
	}

	@Override
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		// Punctuation file.
		if (fileNamePunc != null)
			readerPunc = new BufferedReader(new FileReader(fileNamePunc));

		// Read feature labels in the first line of the file.
		String line = reader.readLine();
		int eq = line.indexOf('=');
		int end = line.indexOf(']');
		String[] labels = line.substring(eq + 1, end).split(",");
		featureLabels = new String[labels.length];
		for (int i = 0; i < labels.length; ++i)
			featureLabels[i] = labels[i].trim();

		// Multi-valued features indexes.
		Set<Integer> multiValuedFeaturesIndexes = new TreeSet<Integer>();
		for (String label : featureLabels)
			multiValuedFeaturesIndexes.add(getFeatureIndex(label));

		// Examples.
		List<DPInput> inputList = new LinkedList<DPInput>();
		List<DPOutput> outputList = new LinkedList<DPOutput>();
		int numExs = 0;
		while (parseExample(reader, multiValuedFeaturesIndexes, "|", inputList,
				outputList)) {
			++numExs;
			if ((numExs + 1) % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}
		System.out.println();
		inputs = inputList.toArray(new DPInput[0]);
		outputs = outputList.toArray(new DPOutput[0]);

		// Close punctuation file.
		if (fileNamePunc != null)
			readerPunc.close();

		LOG.info("Read " + inputs.length + " examples.");
	}

	@Override
	public void save(String fileName) throws IOException, DatasetException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		save(writer);
		writer.close();
	}

	@Override
	public void save(OutputStream os) throws IOException, DatasetException {
		save(new BufferedWriter(new OutputStreamWriter(os)));
	}

	@Override
	public void save(BufferedWriter writer) throws IOException,
			DatasetException {
		throw new NotImplementedException();
	}

	/**
	 * Parse an example from the given line.
	 * 
	 * Each line is a sequence of information chunks separated by TAB
	 * characters. The first chunk is a textual ID of the example. The second
	 * chunk contains only an integer number with the number of tokens in the
	 * sequence, including the root token. The third chunk contains the list of
	 * POS tags of all sequence tokens. From the fourth chunk on, each chunk
	 * comprises list of feature values, separated by space, representing an
	 * edge of the sentence graph.
	 * 
	 * The first feature of each edge comprises its ID that *must* obey the
	 * format head_token_index>dependent_token_index to indicate, respectively,
	 * the start and end points of the edge. The second feature is 1 (one) if
	 * the edge is part of the dependecy tree of the example and 0 (zero)
	 * otherwise. The remaining features represent the edge. Each edge connects
	 * a head token to a dependent token.
	 * 
	 * The edges must come in a specific order in the line. More specifically,
	 * edges are order by dependent token and then by head token. Following this
	 * order, the supposedly first edges should be the ones whose dependent
	 * token is the zero token (root). Since there is no edge incoming to the
	 * root node, this sequence is ommited. Conversely, the diagonal edges
	 * (self-loops), which are also not used in the dependency tree, must not be
	 * ommited but they are ignored. So they can just comprise an arbitrary
	 * character sequence, not including TAB chars.
	 * 
	 * @param reader
	 * @param multiValuedFeatureIndexes
	 * @param valueSeparator
	 * @param inputList
	 * @param outputList
	 * @return
	 * @throws IOException
	 * @throws DatasetException
	 */
	public boolean parseExample(BufferedReader reader,
			Set<Integer> multiValuedFeatureIndexes, String valueSeparator,
			List<DPInput> inputList, List<DPOutput> outputList)
			throws IOException, DatasetException {
		// Read next line.
		String line = reader.readLine();

		if (line == null)
			// End of file.
			return false;

		line = line.trim();
		if (line.length() == 0)
			// Skip consecutive blank lines.
			return true;

		// Skip first line of example, which contains the original sentence.
		String id = readerPunc.readLine();
		line = readerPunc.readLine();
		readerPunc.readLine();

		// Punctuation flags separated by space.
		String[] puncs = REGEX_SPACE.split(line);

		// Number of tokens (including ROOT).
		int numTokens = puncs.length;

		/*
		 * Mark which tokens are considered punctuation and thus are not
		 * considered for evaluation.
		 */
		boolean[] punctuation = new boolean[numTokens];
		for (int idxTkn = 0; idxTkn < numTokens; ++idxTkn)
			punctuation[idxTkn] = puncs[idxTkn].equals("punc");

		/*
		 * List of dependent token edges. Each dependent token is a list of
		 * edges. Each edge is a list of feature codes.
		 */
		ArrayList<ArrayList<LinkedList<Integer>>> features = new ArrayList<ArrayList<LinkedList<Integer>>>(
				numTokens);
		for (int idx = 0; idx < numTokens; ++idx) {
			ArrayList<LinkedList<Integer>> depFtrs = new ArrayList<LinkedList<Integer>>(
					numTokens);
			features.add(depFtrs);
			for (int idxHead = 0; idxHead < numTokens; ++idxHead)
				depFtrs.add(null);
		}

		// Which dependent tokens has the correct edges been seen for.
		boolean[] sawCorrectEdge = new boolean[numTokens];

		// Allocate the output structure.
		DPOutput output = new DPOutput(numTokens);

		// Read next line.
		while ((line = reader.readLine()) != null) {

			line = line.trim();
			if (line.length() == 0)
				// Stop on blank lines.
				break;

			// Split edge in feature values.
			String[] ftrValues = REGEX_SPACE.split(line);

			// Head and dependent tokens indexes.
			String[] edgeId = ftrValues[0].split(">");
			int head = Integer.parseInt(edgeId[0]);
			int dependent = Integer.parseInt(edgeId[1]);

			// Skip diagonal edges.
			if (dependent == head)
				continue;

			// First feature is, in fact, the flag of correct edge.
			String isCorrectEdge = ftrValues[ftrValues.length - 1];

			// If it is the correct edge.
			if (isCorrectEdge.equals("TRUE")) {
				if (sawCorrectEdge[dependent])
					/*
					 * If another correct edge has been seen before, throw an
					 * execption.
					 */
					throw new DatasetException("Double correct edge for token "
							+ dependent);
				output.setHead(dependent, head);
				sawCorrectEdge[dependent] = true;
			} else if (!isCorrectEdge.equals("FALSE")) {
				/*
				 * If it is not the correct edge, but the value is not 0, throw
				 * an exception.
				 */
				throw new DatasetException(
						"Last feature value must be TRUE or FALSE to indicate "
								+ "the correct edge. However, for token "
								+ dependent + " and head " + head
								+ " this feature value is " + isCorrectEdge);
			}

			// Encode the edge features.
			LinkedList<Integer> ftrCodes = new LinkedList<Integer>();
			features.get(dependent).set(head, ftrCodes);
			for (int idxFtr = 0; idxFtr < ftrValues.length; ++idxFtr) {
				String str = ftrValues[idxFtr];
				// TODO deal with multi-valued features.
				int code = basicEncoding.put(new String(str));
				ftrCodes.add(code);
			}
		}

		try {
			/*
			 * Create a new string to store the input id to avoid memory leaks,
			 * since the id string keeps a reference to the line string.
			 */
			DPInput input = new DPInput(new String(id), features, false);
			input.setPunctuation(punctuation);

			// Keep the length of the longest sequence.
			int len = input.getNumberOfTokens();
			if (len > maxNumberOfTokens)
				maxNumberOfTokens = len;

			inputList.add(input);
			outputList.add(output);

			// Return true if there are more lines.
			return line != null;

		} catch (DPInputException e) {
			throw new DatasetException("Error constructing DPInput", e);
		}
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

	@Override
	public FeatureEncoding<String> getFeatureEncoding() {
		return basicEncoding;
	}

	/**
	 * Return the explicit features encoding.
	 * 
	 * @return
	 */
	public MapEncoding<Feature> getExplicitEncoding() {
		return explicitEncoding;
	}

	@Override
	public void serialize(String filename) throws FileNotFoundException,
			IOException {
		throw new NotImplementedException();
	}

	@Override
	public void serialize(String inFilename, String outFilename)
			throws IOException, DatasetException {
		throw new NotImplementedException();
	}

	@Override
	public void deserialize(String filename) throws FileNotFoundException,
			IOException, ClassNotFoundException {
		throw new NotImplementedException();
	}

	//
	// TODO adapt next partition method
	//
	// /**
	// * Store the accumulated weight of each edge for the current template
	// * partition and generate the features for the next partition.
	// *
	// * @return the next partition.
	// */
	// public int nextPartition() {
	// // Input structures.
	// int numExs = inputs.length;
	//
	// // Accumulate current partition feature weights.
	// for (int idxEx = 0; idxEx < numExs; ++idxEx) {
	// DPInput input = inputs[idxEx];
	// int numTkns = input.getNumberOfTokens();
	// for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
	// for (int idxDep = 0; idxDep < numTkns; ++idxDep) {
	// double score = getEdgeScoreFromCurrentFeatures(input,
	// idxHead, idxDep);
	// if (!Double.isNaN(score))
	// fixedWeights[idxEx][idxHead][idxDep] += score;
	// }
	// }
	// }
	//
	// // Go to next partition and generate new features.
	// ++currentPartition;
	// if (currentPartition < numberOfPartitions)
	// generateFeatures();
	// return currentPartition;
	// }

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
			templatesList.add(new SimpleFeatureTemplate(templatesList.size(),
					ftrs));
			// Read next line.
			line = skipBlanksAndComments(reader);
		}

		// Convert list to array.
		numberOfPartitions = 1;
		templates = new FeatureTemplate[1][];
		templates[0] = templatesList.toArray(new FeatureTemplate[0]);

		if (generateFeatures)
			// Generate explicit features.
			generateFeatures();
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
	 * Generate features for the current template partition.
	 */
	public void generateFeatures() {
		LinkedList<Integer> ftrs = new LinkedList<Integer>();
		FeatureTemplate[] tpls = templates[currentPartition];
		int numExs = inputs.length;
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			// Current input structure.
			DPInput input = inputs[idxEx];

			// Number of tokens within the current input.
			int numTkns = input.getNumberOfTokens();

			// Allocate explicit features matrix.
			input.allocFeatureMatrix();

			for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
				for (int idxDep = 0; idxDep < numTkns; ++idxDep) {
					// Skip non-existent edges.
					if (input.getBasicFeatures(idxHead, idxDep) == null)
						continue;

					// Clear previous used list of features.
					ftrs.clear();

					/*
					 * Instantiate edge features and add them to active features
					 * list.
					 */
					for (int idxTpl = 0; idxTpl < tpls.length; ++idxTpl) {
						FeatureTemplate tpl = tpls[idxTpl];
						// Get temporary feature instance.
						Feature ftr = tpl.getInstance(input, idxHead, idxDep);
						// Lookup the feature in the encoding.
						int code = explicitEncoding.getCodeByValue(ftr);
						/*
						 * Instantiate a new feature, if it is not present in
						 * the encoding.
						 */
						if (code == FeatureEncoding.UNSEEN_VALUE_CODE)
							code = explicitEncoding.put(tpl.newInstance(input,
									idxHead, idxDep));
						// Add feature code to active features list.
						ftrs.add(code);
					}

					// Set feature vector of this input.
					input.setFeatures(idxHead, idxDep, ftrs, ftrs.size());
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
}
