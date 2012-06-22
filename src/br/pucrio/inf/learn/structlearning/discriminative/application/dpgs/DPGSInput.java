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
	 * Create an input structure from a list of factors.
	 * 
	 * Each item in this list (i.e., a factor) contains a list of basic
	 * features. The first item in this list contains an array with the factor
	 * parameters. The remaining items are the proper feature values. Each
	 * feature value is an array of values with one or more items.
	 * 
	 * A factor parameter comprises an array with four integers. The first
	 * integer can be 1 or 2 to indicate whether the factor is grandparent or
	 * siblings, respectively. The remaining three integers are the proper
	 * factor paramenter, that is (idxHead, idxModifier, idxGrandparent) for
	 * grandparent factors or (idxHead, idxModifier, idxPreviousModifier) for
	 * siblings factors.
	 * 
	 * @param id
	 *            an arbitrary string that identifies this instance.
	 * @param numberOfColumns
	 *            the number of columns (features) within each factor. This
	 *            number does not include the parameter first feature.
	 * @param basicFeatures
	 *            list of factors comprising parameters and features according
	 *            to the description above.
	 * @throws DPGSException
	 */
	public DPGSInput(String id, int numberOfColumns, int numberOfTokens,
			Collection<? extends Collection<int[]>> basicFeatures)
			throws DPGSException {
		this.id = id;
		this.numberOfTokens = numberOfTokens;
		this.basicGrandparentFeatures = new int[numberOfColumns][numberOfTokens][numberOfTokens][][];
		this.basicSiblingsFeatures = new int[numberOfColumns][numberOfTokens + 1][numberOfTokens + 1][][];

		for (Collection<int[]> factor : basicFeatures) {
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

			/*
			 * Alloc memory for the factor columns and store in the correct
			 * place with the underlying GS structures.
			 */
			switch (params[0]) {
			case 1:
				int idxGrandparent = params[3];
				columns = new int[numberOfColumns][];
				basicGrandparentFeatures[idxHead][idxModifier][idxGrandparent] = columns;
				break;
			case 2:
				int idxPrevModifier = params[3];
				columns = new int[numberOfColumns][];
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

	/**
	 * Create a new grandparent/siblings input structure using the given feature
	 * arrays as underlying features. The given arrays are not copied, that is
	 * they are used as is and must not be modified after this method is called.
	 * 
	 * @param grandparentFeatures
	 * @param siblingsFeatures
	 * @throws DPGSException
	 */
	public DPGSInput(int[][][][] grandparentFeatures,
			int[][][][] siblingsFeatures) throws DPGSException {
		if (grandparentFeatures.length != siblingsFeatures.length)
			throw new DPGSException("Given grandparent feature array has "
					+ "different length of the siblings array");
		this.numberOfTokens = grandparentFeatures.length;
		this.grandparentFeatures = grandparentFeatures;
		this.siblingsFeatures = siblingsFeatures;
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
	 * Return the number of token in this sentence.
	 * 
	 * @return
	 */
	public int size() {
		return numberOfTokens;
	}

}
