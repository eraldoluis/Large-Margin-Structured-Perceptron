package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Collection;
import java.util.Iterator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Bisection input structure. Represent an author and her candidate papers. Each
 * candidate paper is a node of an undirected graph. An arc in this graph
 * represents the relationship between two papers, with features describing how
 * likely these papers are of have being written by the author.
 * 
 * @author eraldo
 * 
 */
public class BisectionInput implements ExampleInput {

	/**
	 * Author id.
	 */
	private long authorId;

	/**
	 * Ids of candidate papers.
	 */
	private long[] papersIds;

	/**
	 * Array of BASIC features (column representation). It is an array of arcs.
	 * The first dimension corresponds to the origin paper (smaller index) and
	 * the second dimension corresponds to the target paper (greater index).
	 * Each arc comprises an fixed-length array of features.
	 */
	private int[][][] basicCategoricalFeatures;

	/**
	 * This input is able to deal with numeric features. In this case, this
	 * array stores these features values.
	 */
	private double[][][] basicNumericalFeatures;

	/**
	 * DERIVED feature codes present in each arc. These features are generated
	 * from templates and there codes are directly used to index model
	 * parameters.
	 */
	private int[][][] featureCodes;

	/**
	 * Derived feature values when numerical features are used within templates.
	 * For features derived from templates composed exclusively by categorical
	 * features, the value is always 1 (one).
	 */
	private double[][][] featureValues;

	/**
	 * Create a new input structure with the given properties.
	 * 
	 * @param authorId
	 *            the author id number.
	 * @param numPapers
	 *            number of candidate papers for this author.
	 * @param categoricalValues
	 *            list of basic features codes. Each item in this list
	 *            corresponds to an edge connecting two candidate papers. An
	 *            edge is a fixed-length list of feature codes (column format).
	 *            The first two items in this list correspond to the indexes of
	 *            the papers connected by the edge. A feature code is the value
	 *            of the categorical feature in some specific position or -1,
	 *            for numerical features.
	 * @param numericalValues
	 *            numeric values for numerical features. For categorical
	 *            features, this value must be Double.NaN.
	 * @throws DatasetException
	 *             in case of misformatted lists (different lenghts, for
	 *             instance).
	 */
	public BisectionInput(int numPapers, long authorId,
			Collection<Long> papersIdsList,
			Collection<? extends Collection<Integer>> categoricalValues,
			Collection<? extends Collection<Double>> numericalValues)
			throws DatasetException {
		// Author ID.
		this.authorId = authorId;

		// Fill papers ids.
		this.papersIds = new long[numPapers];
		int idxPaper = 0;
		for (long paperId : papersIdsList)
			papersIds[idxPaper++] = paperId;

		// Array of basic categorical features.
		this.basicCategoricalFeatures = new int[numPapers][numPapers][];
		// Array of basic numerical features.
		this.basicNumericalFeatures = new double[numPapers][numPapers][];

		// Iterator of edges for categorical features.
		Iterator<? extends Collection<Integer>> itCatEdges = categoricalValues
				.iterator();
		// Iterator of edges for numerical features.
		Iterator<? extends Collection<Double>> itNumEdges = numericalValues
				.iterator();

		int numEdges = categoricalValues.size();
		if (numEdges != numericalValues.size())
			throw new DatasetException(String.format(
					"Author %l has lists of categorical and "
							+ "numerical features with different lengths.",
					authorId));

		int prevNumCatFtrs = Integer.MAX_VALUE;
		int prevNumNumFtrs = Integer.MAX_VALUE;
		int idxEdge = 0;
		while (itCatEdges.hasNext() && itNumEdges.hasNext()) {
			// Read values of categorical features.
			Collection<Integer> catEdge = itCatEdges.next();
			Iterator<Integer> itCatCodes = catEdge.iterator();
			int numCatFtrs = catEdge.size() - 2;

			// Check mininum number of parameters in this edge.
			if (numCatFtrs < 0)
				throw new DatasetException(String.format(
						"Edge in index %d for author %l has less "
								+ "than two features.", idxEdge, authorId));

			// Papers indexes: first two items in the feature list.
			int paper1 = itCatCodes.next();
			int paper2 = itCatCodes.next();

			// Check fixed-length list of features.
			if (prevNumCatFtrs != Integer.MAX_VALUE
					&& prevNumCatFtrs != numCatFtrs)
				throw new DatasetException(String.format(
						"Edge (%d,%d) of author %d has a diferent "
								+ "number of basic features than "
								+ "previous edges.", paper1, paper2, authorId));
			prevNumCatFtrs = numCatFtrs;

			// Allocate and fill array of categorical features for this edge.
			int[] catFtrs = basicCategoricalFeatures[paper1][paper2] = new int[numCatFtrs];
			int idxCatFtr = 0;
			while (itCatCodes.hasNext())
				catFtrs[idxCatFtr++] = itCatCodes.next();

			// Read values of numerical features.
			Collection<Double> numEdge = itNumEdges.next();
			Iterator<Double> itNumVals = numEdge.iterator();
			int numNumFtrs = numEdge.size();

			// Check fixed-length list of features.
			if (prevNumNumFtrs != Integer.MAX_VALUE
					&& prevNumNumFtrs != numNumFtrs)
				throw new DatasetException(String.format(
						"Edge (%d,%d) of author %l has a diferent "
								+ "number of basic numerical features than "
								+ "previous edges.", paper1, paper2, authorId));
			prevNumNumFtrs = numNumFtrs;

			// Allocate and fill array of numerical features for this edge.
			double[] numFtrs = basicNumericalFeatures[paper1][paper2] = new double[numNumFtrs];
			int idxNumFtr = 0;
			while (itNumVals.hasNext())
				numFtrs[idxNumFtr++] = itNumVals.next();

			// Update edge index.
			++idxEdge;
		}

		// Array of derived features.
		this.featureCodes = null;
	}

	@Override
	public String getId() {
		return "" + authorId;
	}

	@Override
	public BisectionOutput createOutput() {
		return new BisectionOutput(basicCategoricalFeatures.length);
	}

	@Override
	public void normalize(double norm) {
		throw new NotImplementedException();
	}

	@Override
	public void sortFeatures() {
		throw new NotImplementedException();
	}

	@Override
	public int getTrainingIndex() {
		throw new NotImplementedException();
	}

	/**
	 * Return the size of this input structure, i.e., the number of items
	 * associated with this query.
	 * 
	 * @return
	 */
	public int size() {
		return basicCategoricalFeatures.length;
	}

	/**
	 * Set the array of derived features with the given collection of
	 * collections of features.
	 * 
	 * @param itemsList
	 */
	public void setFeatures(int paper1, int paper2, Collection<Integer> codes,
			Collection<Double> values) {
		// Number of derived features.
		int numFtrs = codes.size();

		// Allocate arrays for feature codes and values.
		int[] ftrCodes = featureCodes[paper1][paper2] = new int[numFtrs];
		double[] ftrVals = featureValues[paper1][paper2] = new double[numFtrs];

		// Iterate over the codes and values lists and copy their content.
		Iterator<Integer> itCodes = codes.iterator();
		Iterator<Double> itValues = values.iterator();
		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr) {
			ftrCodes[idxFtr] = itCodes.next();
			ftrVals[idxFtr] = itValues.next();
		}
	}

	public int[] getFeatureCodes(int paper1, int paper2) {
		return featureCodes[paper1][paper2];
	}

	public double[] getFeatureValues(int paper1, int paper2) {
		return featureValues[paper1][paper2];
	}

	public int[] getBasicCategoricalFeatures(int paper1, int paper2) {
		return basicCategoricalFeatures[paper1][paper2];
	}

	public double[] getBasicNumericalFeatures(int paper1, int paper2) {
		return basicNumericalFeatures[paper1][paper2];
	}

	/**
	 * Allocate memory for the derived features array
	 */
	public void allocFeatureArray() {
		int numPapers = size();
		featureCodes = new int[numPapers][numPapers][];
		featureValues = new double[numPapers][numPapers][];
	}

	public long getAuthorId() {
		return authorId;
	}

	public long getPaperId(int paper) {
		return papersIds[paper];
	}
}
