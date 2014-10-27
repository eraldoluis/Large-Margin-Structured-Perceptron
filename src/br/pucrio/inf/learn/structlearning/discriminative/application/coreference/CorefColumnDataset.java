package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
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
	 * Indicate whether the parsing algorithm will check multiple incoming edges
	 * tagged as correct for the same token or not.
	 */
	private boolean checkMultipleTrueEdges;

	/**
	 * Create an empty dataset.
	 */
	public CorefColumnDataset() {
		super();
	}

	/**
	 * Create an empty dataset and set the given multi-valued features.
	 * 
	 * @param basicEncoding
	 * @param multiValuedFeatures
	 */
	public CorefColumnDataset(FeatureEncoding<String> basicEncoding,
			Collection<String> multiValuedFeatures) {
		super(basicEncoding, multiValuedFeatures);
		this.checkMultipleTrueEdges = true;
	}

	/**
	 * Create an empty dataset that shares the underlying data structures of the
	 * given dataset.
	 * 
	 * @param sibling
	 */
	public CorefColumnDataset(DPColumnDataset sibling) {
		super(sibling);
		if (sibling instanceof CorefColumnDataset)
			this.checkMultipleTrueEdges = ((CorefColumnDataset) sibling).checkMultipleTrueEdges;
		else
			this.checkMultipleTrueEdges = true;
	}

	/**
	 * Set whether the parsing algorithm will check if multiple incoming edges
	 * are tagged as correct for the same token or not.
	 * 
	 * @param check
	 */
	public void setCheckMultipleTrueEdges(boolean check) {
		this.checkMultipleTrueEdges = check;
	}

	@Override
	protected boolean parseExample(BufferedReader reader,
			Set<Integer> multiValuedFeatureIndexes, String valueSeparator,
			List<DPInput> inputList, List<DPOutput> outputList)
			throws IOException, DatasetException {
		String line;
		int numTokens = -1;
		String id = null;

		// Read punctuation file when given.
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
		LinkedList<Integer> correctRightMentions = new LinkedList<Integer>();
		LinkedList<Integer> correctLeftMentions = new LinkedList<Integer>();

		// Maximum token index.
		int maxIndex = -1;

		// Read next line.
		while ((line = reader.readLine()) != null) {

			line = line.trim();
			if (line.length() == 0)
				// Stop on blank lines.
				break;

			if (line.equals("-"))
				// Empty (no edge) document.
				break;

			// Split edge in feature values.
			String[] ftrValues = REGEX_SPACE.split(line);

			// Head and dependent tokens indexes.
			String[] edgeId = ftrValues[0].split(">");
			int idxMentionLeft = Integer.parseInt(edgeId[0]);
			int idxMentionRight = Integer.parseInt(edgeId[1]);

			// Skip diagonal edges.
			if (idxMentionRight == idxMentionLeft)
				continue;

			if (idxMentionLeft > maxIndex)
				maxIndex = idxMentionLeft;
			if (idxMentionRight > maxIndex)
				maxIndex = idxMentionRight;

			// List of feature codes.
			LinkedList<Integer> edgeFeatures = new LinkedList<Integer>();
			features.add(edgeFeatures);

			// The two first values are the head and the dependent indexes.
			edgeFeatures.add(idxMentionLeft);
			edgeFeatures.add(idxMentionRight);

			/*
			 * The next 6 features indentify the two mentions (their segment and
			 * token indexes). These features are not encoded. They are stored
			 * as their integer values themselves.
			 */
			int m1Segment = Integer.parseInt(ftrValues[1]);
			int m1TokenIni = Integer.parseInt(ftrValues[2]);
			int m1TokenFim = Integer.parseInt(ftrValues[3]);
			int m2Segment = Integer.parseInt(ftrValues[4]);
			int m2TokenIni = Integer.parseInt(ftrValues[5]);
			int m2TokenFim = Integer.parseInt(ftrValues[6]);

			edgeFeatures.add(m1Segment);
			edgeFeatures.add(m1TokenIni);
			edgeFeatures.add(m1TokenFim);
			edgeFeatures.add(m2Segment);
			edgeFeatures.add(m2TokenIni);
			edgeFeatures.add(m2TokenFim);

			// Encode the remaining edge features.
			for (int idxFtr = 7; idxFtr < ftrValues.length - 1; ++idxFtr) {
				String str = ftrValues[idxFtr];
				int code = basicEncoding.put(new String(str));
				edgeFeatures.add(code);
			}

			// The last value is the correct edge flag (TRUE or FALSE).
			String isCorrectEdge = ftrValues[ftrValues.length - 1];
			if (isCorrectEdge.equals("Y")) {
				correctRightMentions.add(idxMentionRight);
				correctLeftMentions.add(idxMentionLeft);
			} else if (!isCorrectEdge.equals("N")) {
				throw new DatasetException(
						"Last feature value must be Y or N to indicate "
								+ "the correct edge. However, for token "
								+ idxMentionRight + " and head "
								+ idxMentionLeft + " this feature value is "
								+ isCorrectEdge);
			}
		}

		if (features.size() == 0 && (line == null || !line.equals("-")))
			return line != null;

		if (numTokens == -1)
			numTokens = maxIndex + 1;

		if (id == null)
			id = "" + inputList.size();

		// Allocate the output structure.
		CorefOutput output = new CorefOutput(numTokens);
		// Fill the output structure.
		Iterator<Integer> itRightMentions = correctRightMentions.iterator();
		Iterator<Integer> itLeftMentions = correctLeftMentions.iterator();
		while (itRightMentions.hasNext() && itLeftMentions.hasNext()) {
			int idxRightMention = itRightMentions.next();
			int idxLeftMention = itLeftMentions.next();
			if (checkMultipleTrueEdges) {
				if (output.getHead(idxRightMention) != -1)
					LOG.warn("Multiple correct incoming edges for token "
							+ idxRightMention + " in example " + id);
				output.setHead(idxRightMention, idxLeftMention);
			} else if (idxLeftMention != 0) {
				output.connectClusters(idxLeftMention, idxRightMention);
			}
		}

		if (checkMultipleTrueEdges)
			// Using mention 0 as the root (artificial mention).
			output.computeClusteringFromTree(0);

		/*
		 * Create a new string to store the input id to avoid memory leaks,
		 * since the id string keeps a reference to the line string.
		 */
		CorefInput input = new CorefInput(numTokens, new String(id), features,
				false);
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
		
		inputs.loadInOrder();

		// Examples.
		for (int idxEx = 0; idxEx < inputs.getNumberExamples(); ++idxEx) {
			DPInput input = (DPInput) inputs.get(idxEx);
			CorefOutput correctOutput = (CorefOutput) outputs[idxEx];
			CorefOutput predictedOutput = (CorefOutput) predictedOuputs[idxEx];

			// Edge features.
			int numMentions = input.getNumberOfTokens();
			for (int idxLeft = 0; idxLeft < numMentions; ++idxLeft) {
				for (int idxRight = 0; idxRight < numMentions; ++idxRight) {
					int[] ftrs = input.getBasicFeatures(idxLeft, idxRight);
					if (ftrs == null)
						continue;

					// Id.
					writer.write(idxLeft + ">" + idxRight);

					/*
					 * The first 6 values are not encoded, they are the integer
					 * values themselver.
					 */
					for (int idxFtr = 0; idxFtr < 6; ++idxFtr)
						writer.write(" " + ftrs[idxFtr]);

					// Ordinary encoded features.
					for (int idxFtr = 6; idxFtr < ftrs.length; ++idxFtr)
						writer.write(" "
								+ basicEncoding.getValueByCode(ftrs[idxFtr]));

					// Correct flag.
					if (correctOutput.getClusterId(idxLeft) == correctOutput
							.getClusterId(idxRight))
						writer.write(" Y");
					else
						writer.write(" N");

					// Predicted feature.
					if (predictedOutput.getClusterId(idxLeft) == predictedOutput
							.getClusterId(idxRight))
						writer.write(" Y");
					else
						writer.write(" N");

					writer.write("\n");
				}
			}

			if (numMentions == 0)
				writer.write("-\n");

			writer.write("\n");
		}
	}

	/**
	 * Save this dataset along with a new column with the given predicted
	 * outputs. However, different of the <code>save(...)</code> methods, save
	 * tag as correct only the edges in the predicted coreference trees.
	 * 
	 * The ordinary <code>save(...)</code> methods tag all intra-cluster edges
	 * as correct. This method is useful to analyse the predicted trees, which
	 * are crucial to understand the prediction algorithm errors.
	 * 
	 * @param fileName
	 * @param predictedOuputs
	 * @param root
	 * @param saveGoldTrees
	 *            if <code>true</code>, in the correct column, tag as Y only
	 *            edges present in the constrained document tree, that is the
	 *            edges that are predicted in the correct outputs. These "gold"
	 *            trees must be predicted before calling this method.
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void saveCorefTrees(String fileName, DPOutput[] predictedOuputs,
			int root, boolean saveGoldTrees) throws IOException,
			DatasetException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
		saveCorefTrees(writer, predictedOuputs, root, saveGoldTrees);
		writer.close();
	}

	/**
	 * Save this dataset along with a new column with the given predicted
	 * outputs. However, different of the <code>save(...)</code> methods, save
	 * tag as correct only the edges in the predicted coreference trees.
	 * 
	 * The ordinary <code>save(...)</code> methods tag all intra-cluster edges
	 * as correct. This method is useful to analyse the predicted trees, which
	 * are crucial to understand the prediction algorithm errors.
	 * 
	 * @param writer
	 * @param predictedOuputs
	 * @param root
	 * @param saveGoldTrees
	 *            if <code>true</code>, in the correct column, tag as Y only
	 *            edges present in the constrained document tree, that is the
	 *            edges that are predicted in the correct outputs. These "gold"
	 *            trees must be predicted before calling this method.
	 * @throws IOException
	 * @throws DatasetException
	 */
	public void saveCorefTrees(BufferedWriter writer,
			DPOutput[] predictedOuputs, int root, boolean saveGoldTrees)
			throws IOException, DatasetException {
		// Header.
		writer.write("[features = id");
		for (int idxFtr = 0; idxFtr < featureLabels.length; ++idxFtr)
			writer.write(", " + featureLabels[idxFtr]);
		writer.write(", correct, predicted]\n\n");

		inputs.loadInOrder();
		
		// Examples.
		for (int idxEx = 0; idxEx < inputs.getNumberExamples(); ++idxEx) {
			DPInput input = (DPInput) inputs.get(idxEx);
			CorefOutput correctOutput = (CorefOutput) outputs[idxEx];
			CorefOutput predictedOutput = (CorefOutput) predictedOuputs[idxEx];

			// Edge features.
			int numMentions = input.getNumberOfTokens();
			for (int idxLeft = 0; idxLeft < numMentions; ++idxLeft) {
				for (int idxRight = 0; idxRight < numMentions; ++idxRight) {
					int[] ftrs = input.getBasicFeatures(idxLeft, idxRight);
					if (ftrs == null)
						continue;
					// Id.
					writer.write(idxLeft + ">" + idxRight);

					/*
					 * The first 6 values are not encoded, they are the integer
					 * values themselver.
					 */
					for (int idxFtr = 0; idxFtr < 6; ++idxFtr)
						writer.write(" " + ftrs[idxFtr]);

					// Features.
					for (int idxFtr = 6; idxFtr < ftrs.length; ++idxFtr)
						writer.write(" "
								+ basicEncoding.getValueByCode(ftrs[idxFtr]));

					// Append correct flag feature.
					if (saveGoldTrees) {
						// Only consider the tree.
						if (idxLeft != root && idxRight != root
								&& correctOutput.getHead(idxRight) == idxLeft)
							writer.write(" Y");
						else
							writer.write(" N");
					} else {
						// Consider the cluster, not the tree.
						if (correctOutput.getClusterId(idxLeft) == correctOutput
								.getClusterId(idxRight))
							writer.write(" Y");
						else
							writer.write(" N");
					}

					// Predicted feature. Only consider the tree.
					if (idxLeft != root && idxRight != root
							&& predictedOutput.getHead(idxRight) == idxLeft)
						writer.write(" Y");
					else
						writer.write(" N");

					writer.write("\n");
				}
			}

			if (numMentions == 0)
				writer.write("-\n");

			writer.write("\n");
		}
	}

}
