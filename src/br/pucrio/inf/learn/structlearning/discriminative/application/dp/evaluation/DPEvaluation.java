package br.pucrio.inf.learn.structlearning.discriminative.application.dp.evaluation;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.AccuracyEvaluation;

/**
 * Evaluation methods for dependency trees.
 * 
 * @author eraldo
 * 
 */
public class DPEvaluation extends AccuracyEvaluation {

	/**
	 * Indicate whether punctuation tokens is considered or not.
	 */
	private boolean includePunctuation;

	/**
	 * Create a dependency parsing evaluation. If
	 * <code>includePunctuation</code> is <code>false</code>, tokens tagged as
	 * punctuation are not accounted in the accuraracy.
	 * 
	 * @param includePunctuation
	 */
	public DPEvaluation(boolean includePunctuation) {
		this.includePunctuation = includePunctuation;
	}

	@Override
	protected int[] evaluateExample(ExampleInput input, ExampleOutput correct,
			ExampleOutput predicted) {
		return evaluateExample((DPInput) input, (DPOutput) correct,
				(DPOutput) predicted);
	}

	/**
	 * Count how many heads are correctly predicted in the given structures.
	 * 
	 * @param input
	 * @param correct
	 * @param predicted
	 * @return a bi-dimensional array where the first element is the number of
	 *         correctly predicted heads and the second element is the total
	 *         number of tokens.
	 */
	protected int[] evaluateExample(DPInput input, DPOutput correct,
			DPOutput predicted) {
		int numTokens = input.getNumberOfTokens();
		int numCorrectTokens = 0;
		int numConsideredTokens = 0;
		for (int idxTkn = 1; idxTkn < numTokens; ++idxTkn) {
			if (!includePunctuation && input.isPunctuation(idxTkn))
				// Skip punctuations if required.
				continue;
			++numConsideredTokens;
			if (correct.getHead(idxTkn) == predicted.getHead(idxTkn))
				++numCorrectTokens;
		}
		return new int[] { numCorrectTokens, numConsideredTokens };
	}

}
