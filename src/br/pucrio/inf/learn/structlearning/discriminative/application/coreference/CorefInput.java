package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import java.util.Collection;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInputException;

/**
 * Represent a coreference resolution input structure. That is a document
 * comprised by a list of mentions and potential links between pairs of
 * mentions. Each link between two mentions comprises a list of features.
 * 
 * @author eraldo
 * 
 */
public class CorefInput extends DPInput {

	/**
	 * Auto-generated serial version UID.
	 */
	private static final long serialVersionUID = -3300651727264367639L;

	/**
	 * Create the input structure of a training example.
	 * 
	 * @param trainingIndex
	 * @para id
	 * @param featuresCollection
	 * 
	 * @throws DPInputException
	 */
	public CorefInput(
			int trainingIndex,
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection)
			throws DPInputException {
		super(trainingIndex, id, featuresCollection);
	}

	/**
	 * Create the input structure of an example that can represent a test
	 * example or just an input structure whose output structure need to be
	 * predicted.
	 * 
	 * @param id
	 * @param featuresCollection
	 * 
	 * @throws DPInputException
	 */
	public CorefInput(
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection)
			throws DPInputException {
		super(id, featuresCollection);
	}

	/**
	 * Create the input structure of an example. The given features are basic
	 * features, that is they come from a column-based dataset.
	 * 
	 * @param id
	 * @param basicFeaturesCollection
	 * @param allocFixedWeightsMatrix
	 * 
	 * @throws DPInputException
	 */
	public CorefInput(
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> basicFeaturesCollection,
			boolean allocFixedWeightsMatrix) throws DPInputException {
		super(id, basicFeaturesCollection, allocFixedWeightsMatrix);
	}

	/**
	 * Create the input structure with a sparse list of features. Each element
	 * in <code>basicFeaturesSparseCollection</code> contains an edge feature
	 * list.
	 * 
	 * The two first values in each edge feature list are the head token index
	 * and the dependent token index. The remaining values are the proper
	 * feature values.
	 * 
	 * @param numberOfTokens
	 * @param id
	 * @param basicFeaturesSparseCollection
	 * @param allocFixedWeightsMatrix
	 */
	public CorefInput(
			int numberOfTokens,
			String id,
			Collection<? extends Collection<Integer>> basicFeaturesSparseCollection,
			boolean allocFixedWeightsMatrix) {
		super(numberOfTokens, id, basicFeaturesSparseCollection,
				allocFixedWeightsMatrix);
	}

	@Override
	public CorefOutput createOutput() {
		return new CorefOutput(getNumberOfTokens());
	}

}
