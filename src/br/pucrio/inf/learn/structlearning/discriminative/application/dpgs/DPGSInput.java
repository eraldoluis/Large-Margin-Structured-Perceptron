package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Input structure for dependency parsing with grandparent and sibling features.
 * 
 * This structure represents a sentence comprised by a sequence of tokens. Each
 * token comprises a fixed array (column-based dataset) of basic features. A
 * feature value can be a single integer or a list of integers (multi-valued
 * features).
 * 
 * Besides input features, this structure stores (grandparent and sibling)
 * factors of derived features, that is, features generated from templates.
 * Derived features are the features used by models to index parameters
 * (weights).
 * 
 * @author eraldo
 * 
 */
public class DPGSInput implements ExampleInput, Serializable {

	/**
	 * Auto-generated serial version ID.
	 */
	private static final long serialVersionUID = 3675190616084423770L;

	/**
	 * Index within the training dataset.
	 */
	private int trainingIndex;

	/**
	 * Textual ID of this example.
	 */
	private String id;

	/**
	 * Number of tokens in this sentence.
	 */
	private int numberOfTokens;

	/**
	 * Basic (column-based) features for edge factors. Such factors are
	 * identified by an index of the form (idxHead, idxModifier).
	 */
	private int[][][][] basicEdgeFeatures;

	/**
	 * Basic (column-based) features for grandparent factors. Each factor is
	 * identified by an index of the form (idxHead, idxModifier,
	 * idxGrandparent). Each factor has a list (columns) of features and each
	 * feature value can comprise one or more values. Most features have only
	 * one value, but there can be some multi-valued features that comprise a
	 * list of values.
	 */
	private int[][][][][] basicGrandparentFeatures;

	/**
	 * Basic (column-based) features for grandparent factors. Each factor is
	 * identified by an index of the form (idxHead, idxModifier,
	 * idxPreviousModifier). Each factor has a list (columns) of features and
	 * each feature value can comprise one or more values. Most features have
	 * only one value, but there can be some multi-valued features that comprise
	 * a list of values.
	 */
	private int[][][][][] basicSiblingsFeatures;

	/**
	 * Derived features from edge templates. These templates involve two
	 * parameters: head and modifier.
	 */
	private int[][][] edgeFeatures;

	/**
	 * Derived features from grandparent templates. These templates involve
	 * three parameters: head, modifier and granparent (head of the head).
	 */
	private int[][][][] grandparentFeatures;

	/**
	 * Derived features from sibling templates. These templates involve three
	 * parameters: head, modifier and closest sibling before the modifier. These
	 * features comprise left and right sibling features. During feature
	 * generation, features are differentiated according to the relative
	 * position of modifiers in relation to the head token.
	 */
	private int[][][][] siblingsFeatures;

	/**
	 * Create an empty input structure with the given length.
	 * 
	 * @param id
	 *            an arbitrary string that identifies this instance.
	 * @param numberOfTokens
	 *            number of tokens in this example.
	 */
	public DPGSInput(String id, int numberOfTokens) {
		this.id = id;
		this.numberOfTokens = numberOfTokens;

		// Arrays for basic features.
		this.basicEdgeFeatures = new int[numberOfTokens][numberOfTokens][][];
		this.basicGrandparentFeatures = new int[numberOfTokens][numberOfTokens][numberOfTokens][][];
		this.basicSiblingsFeatures = new int[numberOfTokens][numberOfTokens + 1][numberOfTokens + 1][][];

		// Arrays for derived features.
		this.edgeFeatures = new int[numberOfTokens][numberOfTokens][];
		this.grandparentFeatures = new int[numberOfTokens][numberOfTokens][numberOfTokens][];
		this.siblingsFeatures = new int[numberOfTokens][numberOfTokens + 1][numberOfTokens + 1][];
	}

	/**
	 * Create a new grandparent/siblings input structure using the given feature
	 * arrays as underlying features. The given arrays are not copied, that is
	 * they are used as is and must not be modified after this method is called.
	 * 
	 * @param edgeFeatures
	 * @param grandparentFeatures
	 * @param siblingsFeatures
	 * @throws DPGSException
	 */
	public DPGSInput(int[][][] edgeFeatures, int[][][][] grandparentFeatures,
			int[][][][] siblingsFeatures) throws DPGSException {
		// All three arrays must have the same dimension.
		if (grandparentFeatures.length != siblingsFeatures.length
				|| edgeFeatures.length != siblingsFeatures.length)
			throw new DPGSException("Feature array have different length");

		this.numberOfTokens = grandparentFeatures.length;
		this.edgeFeatures = edgeFeatures;
		this.grandparentFeatures = grandparentFeatures;
		this.siblingsFeatures = siblingsFeatures;
	}

	/**
	 * Add basic features of the given list of factors.
	 * 
	 * Each item in this list (i.e., a factor) contains a list of basic
	 * features. The first item in this list contains an array with the factor
	 * parameters. The remaining items are the proper feature values. Each
	 * feature value is an array of values with one or more items.
	 * 
	 * A factor parameter comprises an array with four integers. The first
	 * integer indicates its type and can be 0 (edge factor), 1 (grandparent) or
	 * 2 (siblings).
	 * 
	 * The remaining three integers are the proper factor parameters. For
	 * grandparent factors, (idxHead, idxModifier, idxGrandparent). For siblings
	 * factors (idxHead, idxModifier, idxPreviousModifier). For edge factors it
	 * is (idxHead, idxModifier) and the third parameter is irrelevant.
	 * 
	 * @param factors
	 *            list of factors to be included in this input structure.
	 * @throws DPGSException
	 */
	public void addBasicFeaturesOfFactors(
			Collection<? extends Collection<int[]>> factors)
			throws DPGSException {
		for (Collection<int[]> factor : factors) {
			// Iterator of factor list items.
			Iterator<int[]> it = factor.iterator();

			/*
			 * The first item contains the factor parameters: (type, idxHead,
			 * idxModifier, idxGrandparent/idxPrevModifier).
			 */
			int[] params = it.next();

			/*
			 * For both types of factor, the second and third parameter are the
			 * head index and the modifier index.
			 */
			int idxHead = params[1];
			int idxModifier = params[2];

			// Columns for the current factor.
			int[][] columns;

			// Number of basic features in this factor.
			int numberOfColumns = factor.size() - 1;

			/*
			 * Alloc memory for the factor columns and store in the correct
			 * place with the underlying GS structures.
			 */
			switch (params[0]) {
			case 0:
				// EDGE factor.
				columns = new int[numberOfColumns][];
				if (basicEdgeFeatures[idxHead][idxModifier] != null)
					throw new DPGSException(
							String.format(
									"Factor E(%d,%d,%d) in example %s is already filled",
									idxHead, idxModifier, params[3], id));
				basicEdgeFeatures[idxHead][idxModifier] = columns;
				break;
			case 1:
				// GRANDPARENT factor.
				int idxGrandparent = params[3];
				columns = new int[numberOfColumns][];
				if (basicGrandparentFeatures[idxHead][idxModifier][idxGrandparent] != null)
					throw new DPGSException(
							String.format(
									"Factor G(%d,%d,%d) in example %s is already filled",
									idxHead, idxModifier, idxGrandparent, id));
				basicGrandparentFeatures[idxHead][idxModifier][idxGrandparent] = columns;
				break;
			case 2:
				// SIBLINGS factor.
				int idxPrevModifier = params[3];
				columns = new int[numberOfColumns][];
				if (basicSiblingsFeatures[idxHead][idxModifier][idxPrevModifier] != null)
					throw new DPGSException(
							String.format(
									"Factor S(%d,%d,%d) in example %s is already filled",
									idxHead, idxModifier, idxPrevModifier, id));
				basicSiblingsFeatures[idxHead][idxModifier][idxPrevModifier] = columns;
				break;
			default:
				throw new DPGSException(
						String.format(
								"Incorrect factor type %d in example %s with params (%d,%d,%d)",
								params[0], id, params[1], params[2], params[3]));
			}

			// Just fill the columns array.
			int idx = 0;
			while (it.hasNext())
				columns[idx++] = it.next();

			if (idx != columns.length)
				throw new DPGSException(
						String.format(
								"Incorrect number of features in the factor (%d,%d,%d) of type %d from example %s",
								params[1], params[2], params[3], params[0], id));
		}
	}

	@Override
	public String getId() {
		return id;
	}

	@Override
	public int getTrainingIndex() {
		return trainingIndex;
	}

	/**
	 * Return the list of feature codes in the given edge factor (idxHead,
	 * idxModifier).
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @return
	 */
	public int[] getEdgeFeatures(int idxHead, int idxModifier) {
		return edgeFeatures[idxHead][idxModifier];
	}

	/**
	 * Return the list of feature codes in the given grandparent factor.
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param idxGrandparent
	 * @return
	 */
	public int[] getGrandparentFeatures(int idxHead, int idxModifier,
			int idxGrandparent) {
		return grandparentFeatures[idxHead][idxModifier][idxGrandparent];
	}

	/**
	 * Return the list of feature codes in the given modifiers factor.
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param idxSibling
	 * @return
	 */
	public int[] getSiblingsFeatures(int idxHead, int idxModifier,
			int idxSibling) {
		return siblingsFeatures[idxHead][idxModifier][idxSibling];
	}

	public void setEdgeFeatures(int idxHead, int idxModifier, int[] vals) {
		edgeFeatures[idxHead][idxModifier] = vals;
	}

	public void setGrandparentFeatures(int idxHead, int idxModifier,
			int idxGrandparent, int[] vals) {
		grandparentFeatures[idxHead][idxModifier][idxGrandparent] = vals;
	}

	public void setSiblingsFeatures(int idxHead, int idxModifier,
			int idxPrevModifier, int[] vals) {
		siblingsFeatures[idxHead][idxModifier][idxPrevModifier] = vals;
	}

	@Override
	public DPGSOutput createOutput() {
		return new DPGSOutput(numberOfTokens);
	}

	@Override
	public void normalize(double norm) {
		throw new NotImplementedException();
	}

	@Override
	public void sortFeatures() {
		throw new NotImplementedException();
	}

	/**
	 * Return the number of tokens in this sentence.
	 * 
	 * @return
	 */
	public int size() {
		return numberOfTokens;
	}

	/**
	 * Return the column-based features for the given edge factor (idxHead,
	 * idxModifier).
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @return
	 */
	public int[][] getBasicEdgeFeatures(int idxHead, int idxModifier) {
		return basicEdgeFeatures[idxHead][idxModifier];
	}

	/**
	 * Return the column-based features for the given grandparent factor
	 * (idxHead, idxModifier, idxGrandparent).
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param idxGrandparent
	 * @return
	 */
	public int[][] getBasicGrandparentFeatures(int idxHead, int idxModifier,
			int idxGrandparent) {
		return basicGrandparentFeatures[idxHead][idxModifier][idxGrandparent];
	}

	/**
	 * Return the column-based features for the given siblings factor (idxHead,
	 * idxModifier, idxSibling).
	 * 
	 * @param idxHead
	 * @param idxModifier
	 * @param idxPrevModifier
	 * @return
	 */
	public int[][] getBasicSiblingsFeatures(int idxHead, int idxModifier,
			int idxPrevModifier) {
		return basicSiblingsFeatures[idxHead][idxModifier][idxPrevModifier];
	}
}
