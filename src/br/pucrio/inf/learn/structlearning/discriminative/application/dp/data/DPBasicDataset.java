package br.pucrio.inf.learn.structlearning.discriminative.application.dp.data;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.SimpleExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

/**
 * Represent a sequence of dependency parsing examples.
 * 
 * @author eraldo
 * 
 */
public class DPBasicDataset implements DPDataset {

	/**
	 * Logging object.
	 */
	private static Log LOG = LogFactory.getLog(DPBasicDataset.class);

	/**
	 * Regular expression pattern to parse spaces.
	 */
	private final Pattern REGEX_SPACE = Pattern.compile("[ ]");

	/**
	 * Regular expression pattern to parse TAB characters.
	 */
	private final Pattern REGEX_TAB = Pattern.compile("\\t");

	/**
	 * Textual features encoding scheme.
	 */
	private FeatureEncoding<String> encoding;

	/**
	 * Input sequences.
	 */
	private ExampleInputArray inputs;

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
	 * Default constructor. Create an empty dataset with the default encoding
	 * scheme: <code>StringMapEncoding</code>.
	 */
	public DPBasicDataset() {
		encoding = new StringMapEncoding();
	}

	/**
	 * Create a dataset using an existing feature encoding.
	 * 
	 * @param encoding
	 */
	public DPBasicDataset(FeatureEncoding<String> encoding) {
		this.encoding = encoding;
	}

	@Override
	public ExampleInputArray getInputs() {
		return inputs;
	}

	@Override
	public DPOutput[] getOutputs() {
		return outputs;
	}

	@Override
	public DPInput getInput(int index) {
		inputs.load(new int[index]);
		return (DPInput) inputs.get(index);
	}

	@Override
	public DPOutput getOutput(int index) {
		return outputs[index];
	}

	@Override
	public int getNumberOfExamples() {
		return inputs.getNumberExamples();
	}

	@Override
	public boolean isTraining() {
		return training;
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

	@Override
	public void load(BufferedReader reader) throws IOException,
			DatasetException {
		List<DPInput> inputList = new LinkedList<DPInput>();
		List<DPOutput> outputList = new LinkedList<DPOutput>();
		int numExs = 0;
		while (parseExample(reader, inputList, outputList)) {
			++numExs;
			if (numExs % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}
		System.out.println();
		inputs = new SimpleExampleInputArray(inputList.size());
		
		for (DPInput dpInput : inputList) {
			inputs.put(dpInput);
		}
		
		outputs = outputList.toArray(new DPOutput[0]);

		LOG.info("Read " + inputs.getNumberExamples() + " examples.");
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
	 * @param inputList
	 * @param outputList
	 * @return
	 * @throws IOException
	 * @throws DatasetException
	 */
	public boolean parseExample(BufferedReader reader, List<DPInput> inputList,
			List<DPOutput> outputList) throws IOException, DatasetException {
		// Read next line.
		String line = reader.readLine();

		if (line == null)
			// End of file.
			return false;

		line = line.trim();
		if (line.length() == 0)
			// Skip consecutive blank lines.
			return true;

		// Sequence features, separated by TAB.
		String[] seqFtrs = REGEX_TAB.split(line);

		// First chunk is the example ID.
		String id = seqFtrs[0];

		// Second chunk contains only the number of tokens.
		int numTokens = Integer.parseInt(seqFtrs[1]);

		/*
		 * Parse the pos tags of each token (third chunk) and keep which ones
		 * are tagged as punctuation.
		 */
		boolean[] punctuation = new boolean[numTokens];
		String[] posTags = REGEX_SPACE.split(seqFtrs[2]);
		for (int idxTkn = 0; idxTkn < numTokens; ++idxTkn)
			if (posTags[idxTkn].equals("punc")
					|| posTags[idxTkn].equals("##ROOT##"))
				punctuation[idxTkn] = true;

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
			int isCorrectEdge = Integer.parseInt(ftrValues[1]);

			// If it is the correct edge.
			if (isCorrectEdge == 1) {
				if (sawCorrectEdge[dependent])
					/*
					 * If another correct edge has been seen before, throw an
					 * execption.
					 */
					throw new DatasetException("Double correct edge for token "
							+ dependent);
				output.setHead(dependent, head);
				sawCorrectEdge[dependent] = true;
			} else if (isCorrectEdge != 0) {
				/*
				 * If it is not the correct edge, but the value is not 0, throw
				 * an exception.
				 */
				throw new DatasetException(
						"First feature value must be 0 or 1 to indicate "
								+ "the correct edge. However, this is not "
								+ "true for token " + dependent + " and head "
								+ head);
			}

			// Encode the edge features.
			LinkedList<Integer> ftrCodes = new LinkedList<Integer>();
			features.get(dependent).set(head, ftrCodes);
			for (int idxFtr = 2; idxFtr < ftrValues.length; ++idxFtr) {
				/*
				 * Create a new string before including the feature value in the
				 * encoding, since the ftrValues[idxFtr] string keeps a
				 * reference to the line string.
				 */
				int code = encoding.put(new String(ftrValues[idxFtr]));
				if (code >= 0)
					ftrCodes.add(code);
			}
		}

		try {
			/*
			 * Create a new string to store the input id to avoid memory leaks,
			 * since the id string keeps a reference to the line string.
			 */
			DPInput input = new DPInput(new String(id), features);
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

	@Override
	public void serialize(String filename) throws FileNotFoundException,
			IOException {
		ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(
				filename));
		os.writeObject(inputs);
		os.writeObject(outputs);
		os.writeBoolean(training);
		os.writeInt(maxNumberOfTokens);
		os.close();
	}

	@Override
	public void serialize(String inFilename, String outFilename)
			throws IOException, DatasetException {
		BufferedReader reader = new BufferedReader(new FileReader(inFilename));
		ObjectOutputStream os = new ObjectOutputStream(
				new BufferedOutputStream(new FileOutputStream(outFilename)));
		List<DPInput> inputList = new LinkedList<DPInput>();
		List<DPOutput> outputList = new LinkedList<DPOutput>();
		int numExs = 0;
		while (parseExample(reader, inputList, outputList)) {
			if (inputList.size() > 0) {
				DPInput input = inputList.remove(0);
				DPOutput output = outputList.remove(0);
				os.writeObject(input);
				os.writeObject(output);
				++numExs;
				if (numExs % 100 == 0) {
					System.out.print(".");
					System.out.flush();
				}
			}
		}
		os.writeObject(new Boolean(training));
		os.writeObject(new Integer(maxNumberOfTokens));
		os.close();
		reader.close();
		LOG.info("Read " + numExs + " examples.");
	}

	@Override
	public void deserialize(String filename) throws FileNotFoundException,
			IOException, ClassNotFoundException {
		ObjectInputStream is = new ObjectInputStream(new BufferedInputStream(
				new FileInputStream(filename)));
		List<DPInput> inputList = new LinkedList<DPInput>();
		List<DPOutput> outputList = new LinkedList<DPOutput>();
		int numExs = 0;
		Object obj;
		while ((obj = is.readObject()) != null) {
			if (!(obj instanceof DPInput))
				break;
			inputList.add((DPInput) obj);
			outputList.add((DPOutput) is.readObject());
			++numExs;
			if (numExs % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}
		System.out.println();
		inputs = new SimpleExampleInputArray(inputList.size());
		
		for (DPInput dpInput : inputList) {
			try {
				inputs.put(dpInput);
			} catch (DatasetException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		outputs = outputList.toArray(new DPOutput[0]);
		training = (Boolean) obj;
		maxNumberOfTokens = (Integer) is.readObject();
		is.close();
	}

}
