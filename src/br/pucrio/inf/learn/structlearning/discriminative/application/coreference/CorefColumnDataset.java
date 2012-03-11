package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;

/**
 * Store a coreference dataset in column format and provide methods related to
 * feature template manipulation.
 * 
 * @author eraldo
 * 
 */
public class CorefColumnDataset extends DPColumnDataset {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(CorefColumnDataset.class);

	/**
	 * Create an empty dataset and set the given multi-valued features.
	 * 
	 * @param basicEncoding
	 * @param multiValuedFeatures
	 */
	public CorefColumnDataset(FeatureEncoding<String> basicEncoding,
			Collection<String> multiValuedFeatures) {
		super(basicEncoding, multiValuedFeatures);
	}

	/**
	 * Create an empty dataset that shares the underlying data structures of the
	 * given dataset.
	 * 
	 * @param sibling
	 */
	public CorefColumnDataset(DPColumnDataset sibling) {
		super(sibling);
	}

	@Override
	protected boolean parseExample(BufferedReader reader,
			Set<Integer> multiValuedFeatureIndexes, String valueSeparator,
			List<DPInput> inputList, List<DPOutput> outputList)
			throws IOException, DatasetException {
		// Skip first line of example, which contains the original sentence.
		String line;
		int numTokens = -1;
		String id = null;
		String[] puncs = null;
		boolean[] punctuation = null;
		if (readerPunc != null) {
			id = readerPunc.readLine();
			line = readerPunc.readLine();
			if (line != null) {
				readerPunc.readLine();
				// Punctuation flags separated by space.
				puncs = REGEX_SPACE.split(line);

				/*
				 * Mark which tokens are considered punctuation and thus are not
				 * considered for evaluation.
				 */
				numTokens = puncs.length;
				punctuation = new boolean[numTokens];
				for (int idxTkn = 0; idxTkn < numTokens; ++idxTkn)
					punctuation[idxTkn] = puncs[idxTkn].equals("punc");
			}
		}

		/*
		 * List of edges. Each edge is a list of feature codes. However, the two
		 * first values in each list are the head token index and the dependent
		 * token index, and the third value is 1, if the edge is correct, and 0,
		 * otherwise.
		 */
		LinkedList<LinkedList<Integer>> features = new LinkedList<LinkedList<Integer>>();

		// Correct edges (head-dependent pairs).
		LinkedList<Integer> correctDepTokens = new LinkedList<Integer>();
		LinkedList<Integer> correctHeadTokens = new LinkedList<Integer>();

		// Maximum token index.
		int maxIndex = -1;

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
			int idxHead = Integer.parseInt(edgeId[0]);
			int idxDep = Integer.parseInt(edgeId[1]);

			// Skip diagonal edges.
			if (idxDep == idxHead)
				continue;

			if (idxHead > maxIndex)
				maxIndex = idxHead;
			if (idxDep > maxIndex)
				maxIndex = idxDep;

			// List of feature codes.
			LinkedList<Integer> edgeFeatures = new LinkedList<Integer>();
			features.add(edgeFeatures);

			// The two first values are the head and the dependent indexes.
			edgeFeatures.add(idxHead);
			edgeFeatures.add(idxDep);

			// Encode the edge features.
			for (int idxFtr = 1; idxFtr < ftrValues.length - 1; ++idxFtr) {
				String str = ftrValues[idxFtr];
				// TODO deal with multi-valued features.
				int code = basicEncoding.put(new String(str));
				edgeFeatures.add(code);
			}

			// The last value is the correct edge flag (TRUE or FALSE).
			String isCorrectEdge = ftrValues[ftrValues.length - 1];
			if (isCorrectEdge.equals("Y")) {
				correctDepTokens.add(idxDep);
				correctHeadTokens.add(idxHead);
			} else if (!isCorrectEdge.equals("N")) {
				/*
				 * If it is not the correct edge, but the value is not 0, throw
				 * an exception.
				 */
				throw new DatasetException(
						"Last feature value must be TRUE or FALSE to indicate "
								+ "the correct edge. However, for token "
								+ idxDep + " and head " + idxHead
								+ " this feature value is " + isCorrectEdge);
			}
		}

		if (features.size() == 0)
			return line != null;

		if (numTokens == -1)
			numTokens = maxIndex + 1;

		if (id == null)
			id = "" + inputList.size();

		// Allocate the output structure.
		DPOutput output = new DPOutput(numTokens);
		// Fill the output structure.
		Iterator<Integer> itDep = correctDepTokens.iterator();
		Iterator<Integer> itHead = correctHeadTokens.iterator();
		while (itDep.hasNext() && itHead.hasNext()) {
			int idxDep = itDep.next();
			int idxHead = itHead.next();
			if (output.getHead(idxDep) != -1)
				LOG.warn("Multiple correct incoming edges for token " + idxDep
						+ " in example " + id);
			output.setHead(idxDep, idxHead);
		}

		/*
		 * Create a new string to store the input id to avoid memory leaks,
		 * since the id string keeps a reference to the line string.
		 */
		DPInput input = new DPInput(numTokens, new String(id), features, false);
		if (punctuation != null)
			input.setPunctuation(punctuation);

		// Keep the length of the longest sequence.
		int len = input.getNumberOfTokens();
		if (len > maxNumberOfTokens)
			maxNumberOfTokens = len;

		inputList.add(input);
		outputList.add(output);

		// Return true if there are more lines.
		return line != null;
	}

	/**
	 * Save this dataset along with a new column with the given predicted
	 * outputs.
	 * 
	 * @param writer
	 * @param predictedOuputs
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void save(BufferedWriter writer, DPOutput[] predictedOuputs)
			throws IOException, DatasetException {
		// Header.
		writer.write("[features = id");
		for (int idxFtr = 0; idxFtr < featureLabels.length; ++idxFtr)
			writer.write(", " + featureLabels[idxFtr]);
		writer.write(", correct, predicted]\n\n");

		// Examples.
		for (int idxEx = 0; idxEx < inputs.length; ++idxEx) {
			DPInput input = inputs[idxEx];
			DPOutput correctOutput = outputs[idxEx];
			DPOutput predictedOutput = predictedOuputs[idxEx];

			// Edge features.
			int numTokens = input.getNumberOfTokens();
			for (int idxDep = 1; idxDep < numTokens; ++idxDep) {
				for (int idxHead = 1; idxHead < numTokens; ++idxHead) {
					int[] ftrs = input.getBasicFeatures(idxHead, idxDep);
					if (ftrs == null)
						continue;
					// Id.
					writer.write(idxHead + ">" + idxDep);
					// Features.
					for (int idxFtr = 0; idxFtr < ftrs.length; ++idxFtr)
						writer.write(" "
								+ basicEncoding.getValueByCode(ftrs[idxFtr]));

					// Correct feature.
					if (correctOutput.getHead(idxDep) == idxHead)
						writer.write(" Y");
					else
						writer.write(" N");

					// Predicted feature.
					if (predictedOutput.getHead(idxDep) == idxHead)
						writer.write(" Y");
					else
						writer.write(" N");

					writer.write("\n");
				}
			}

			writer.write("\n");
		}
	}
	
}
