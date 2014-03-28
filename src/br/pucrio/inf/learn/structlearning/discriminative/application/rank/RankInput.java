package br.pucrio.inf.learn.structlearning.discriminative.application.rank;

import java.util.Collection;
import java.util.Iterator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;

/**
 * Ranking input structure. It prepresents a query and a list of items
 * (documents). Each item comprises an id and a list of features.
 * 
 * @author eraldo
 * 
 */
public class RankInput implements ExampleInput {

	/**
	 * Query id.
	 */
	private long queryId;

	/**
	 * Array of basic features (column representation). It is an array of items.
	 * Each item comprises an array of features, which has a fixed length.
	 */
	private int[][] basicFeatures;

	/**
	 * Derived features for each item. These features are generated from
	 * templates and the ones directly used within models.
	 */
	private int[][] features;

	/**
	 * Create a new rank input using the given query ID and the list of items
	 * features.
	 * 
	 * @param queryId
	 * @param items
	 * @throws DatasetException
	 */
	public RankInput(long queryId,
			Collection<? extends Collection<Integer>> items)
			throws DatasetException {
		// Query ID.
		this.queryId = queryId;

		// Array of items basic features.
		this.basicFeatures = new int[items.size()][];
		Iterator<? extends Collection<Integer>> itItems = items.iterator();
		int item = 0;
		int prevNumFtrs = Integer.MAX_VALUE;
		while (itItems.hasNext()) {
			Collection<Integer> ftrs = itItems.next();
			Iterator<Integer> itFtrs = ftrs.iterator();
			int idxFtr = 0;
			int numFtrs = ftrs.size();
			if (prevNumFtrs != Integer.MAX_VALUE && prevNumFtrs != numFtrs)
				throw new DatasetException(
						String.format(
								"Item %d in query %l has a diferent number of basic features than previous items.",
								item, queryId));
			prevNumFtrs = numFtrs;
			basicFeatures[item] = new int[numFtrs];
			while (itFtrs.hasNext()) {
				basicFeatures[item][idxFtr] = itFtrs.next();
				++idxFtr;
			}
			++item;
		}

		// Array of derived features.
		this.features = null;
	}

	@Override
	public String getId() {
		return "" + queryId;
	}

	@Override
	public RankOutput createOutput() {
		return new RankOutput(basicFeatures.length);
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
		return basicFeatures.length;
	}

	/**
	 * Set the array of derived features with the given collection of
	 * collections of features.
	 * 
	 * @param itemsList
	 */
	public void setFeatures(int item, Collection<Integer> featuresList) {
		int numFtrs = featuresList.size();
		features[item] = new int[numFtrs];
		Iterator<Integer> it = featuresList.iterator();
		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr)
			features[item][idxFtr] = it.next().intValue();
	}

	/**
	 * Return the derived features array of the given item.
	 * 
	 * @param item
	 * @return
	 */
	public int[] getFeatures(int item) {
		return features[item];
	}

	/**
	 * Return the basic features array for the given item.
	 * 
	 * @param item
	 * @return
	 */
	public int[] getBasicFeatures(int item) {
		return basicFeatures[item];
	}

	/**
	 * Allocate memory for the derived features array
	 */
	public void allocFeatureArray() {
		features = new int[size()][];
	}
	
	public long getQueryId() {
		return queryId;
	}
}
