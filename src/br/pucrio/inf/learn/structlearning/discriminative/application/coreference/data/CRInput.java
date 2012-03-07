package br.pucrio.inf.learn.structlearning.discriminative.application.coreference.data;

import java.util.Collection;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInputException;

/**
 * Input structure of a dependency parsing example. Represent a complete
 * directed graph whose nodes are tokens of a sentence. An edge is composed by a
 * list of features between two tokens in the sentence.
 * 
 * @author eraldo
 * 
 */
public class CRInput extends DPInput {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1673097356834407574L;

	/**
	 * Create the input structure of a training example.
	 * 
	 * @param trainingIndex
	 * @param featuresCollection
	 * @throws DPInputException
	 */
	public CRInput(
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
	 * @param featuresCollection
	 * @throws DPInputException
	 */
	public CRInput(
			String id,
			Collection<? extends Collection<? extends Collection<Integer>>> featuresCollection)
			throws DPInputException {
		super(id, featuresCollection);
	}

}
