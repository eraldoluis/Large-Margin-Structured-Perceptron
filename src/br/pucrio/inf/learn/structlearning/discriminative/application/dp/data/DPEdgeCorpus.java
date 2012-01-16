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
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a dataset with dependency parsing examples. Each edge within a
 * sentence is represented by a fixed number of features. This class is useful
 * for template-based models.
 * 
 * @author eraldo
 * 
 */
public class DPEdgeCorpus implements DPDataset {

	/**
	 * Logging object.
	 */
	private static Log LOG = LogFactory.getLog(DPEdgeCorpus.class);

	/**
	 * Regular expression pattern to parse spaces.
	 */
	private final Pattern REGEX_SPACE = Pattern.compile("[ ]");

	/**
	 * Encoding for basic textual features.
	 */
	private FeatureEncoding<String> encoding;

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
	 * Explicit lists of features to speedup some algorithms. Mainly, extraction
	 * algorithms for small corpus (testing data, for instance).
	 */
	@SuppressWarnings("rawtypes")
	private LinkedList[][][] explicitFeatures;

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
	public DPEdgeCorpus(Collection<String> multiValuedFeatures) {
		this.multiValuedFeatures = new TreeSet<String>();
		if (multiValuedFeatures != null) {
			for (String ftrLabel : multiValuedFeatures)
				this.multiValuedFeatures.add(ftrLabel);
		}
		this.encoding = new StringMapEncoding();
	}

	/**
	 * Create edge corpus with the given encoding.
	 * 
	 * @param multiValuedFeatures
	 * @param encoding
	 */
	public DPEdgeCorpus(Collection<String> multiValuedFeatures,
			FeatureEncoding<String> encoding) {
		this.multiValuedFeatures = new TreeSet<String>();
		if (multiValuedFeatures != null) {
			for (String ftrLabel : multiValuedFeatures)
				this.multiValuedFeatures.add(ftrLabel);
		}
		this.encoding = encoding;
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
	 * Return the lists of explicit features.
	 * 
	 * These lists can be used to speedup extraction algorithms for small corpus
	 * (testing data, for instance).
	 * 
	 * @return
	 */
	@SuppressWarnings("rawtypes")
	public LinkedList[][][] getExplicitFeatures() {
		return explicitFeatures;
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
	 * Create lists of explicit features for this corpus.
	 * 
	 * @param templates
	 */
	public void createExplicitFeatures(FeatureTemplate[] templates) {
		int numExs = getNumberOfExamples();
		explicitFeatures = new LinkedList[numExs][][];
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			DPInput input = inputs[idxEx];
			int lenEx = input.getNumberOfTokens();
			explicitFeatures[idxEx] = new LinkedList[lenEx][lenEx];
			for (int head = 0; head < lenEx; ++head) {
				for (int dependent = 0; dependent < lenEx; ++dependent) {
					if (input.getFeatureCodes(head, dependent) == null) {
						// Inexistent edge.
						explicitFeatures[idxEx][head][dependent] = null;
						continue;
					}

					// Explicitly generate edge features from templates.
					LinkedList<Feature> features = new LinkedList<Feature>();
					explicitFeatures[idxEx][head][dependent] = features;
					for (int idxTpl = 0; idxTpl < templates.length; ++idxTpl) {
						FeatureTemplate template = templates[idxTpl];
						Feature ftr = template.newInstance(input, head,
								dependent, idxTpl);
						features.add(ftr);
					}
				}
			}

			// Progress info.
			if ((idxEx + 1) % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}
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
				int code = encoding.put(new String(str));
				ftrCodes.add(code);
			}
		}

		try {
			/*
			 * Create a new string to store the input id to avoid memory leaks,
			 * since the id string keeps a reference to the line string.
			 */
			DPInput input = new DPInput(inputList.size(), new String(id),
					features);
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

	/**
	 * @return the feature encoding used by this dataset.
	 */
	public FeatureEncoding<String> getFeatureEncoding() {
		return encoding;
	}

	/**
	 * Load templates from the given reader.
	 * 
	 * @param reader
	 * @return
	 * @throws IOException
	 */
	public FeatureTemplate[] loadTemplates(BufferedReader reader)
			throws IOException {
		LinkedList<FeatureTemplate> templates = new LinkedList<FeatureTemplate>();
		String line = skipBlanksAndComments(reader);
		while (line != null) {
			String[] ftrsStr = REGEX_SPACE.split(line);
			int[] ftrs = new int[ftrsStr.length];
			for (int idx = 0; idx < ftrs.length; ++idx)
				ftrs[idx] = getFeatureIndex(ftrsStr[idx]);
			templates.add(new SimpleFeatureTemplate(templates.size(), ftrs));
			// Read next line.
			line = skipBlanksAndComments(reader);
		}
		return templates.toArray(new FeatureTemplate[0]);
	}

	/**
	 * Load templates from the given file.
	 * 
	 * @param templatesFileName
	 * @return
	 * @throws IOException
	 */
	public FeatureTemplate[] loadTemplates(String templatesFileName)
			throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		FeatureTemplate[] templates = loadTemplates(reader);
		reader.close();
		return templates;
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
}
