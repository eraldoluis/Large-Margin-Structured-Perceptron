package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation;

import java.util.Collection;

import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.EntityF1Evaluation;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.TypedEntity;

public class LabeledTokenEvaluation extends EntityF1Evaluation {

	/**
	 * Encode the label strings.
	 */
	private FeatureEncoding<String> labelEncoding;

	/**
	 * Construct a new evaluation object with the given label encoding.
	 * 
	 * @param labelEncoding
	 */
	public LabeledTokenEvaluation(FeatureEncoding<String> labelEncoding) {
		this.labelEncoding = labelEncoding;
	}

	@Override
	public void decodeEntities(ExampleInput input, ExampleOutput output,
			Collection<TypedEntity> entities) {
		SequenceOutput seqOutput = (SequenceOutput) output;
		for (int tkn = 0; tkn < seqOutput.size(); ++tkn)
			entities.add(new LabeledToken(labelEncoding
					.getValueByCode(seqOutput.getLabel(tkn)), tkn));
	}

	@Override
	public void encodeEntity(ExampleInput input, ExampleOutput output,
			TypedEntity entity) {
		SequenceOutput seqOutput = (SequenceOutput) output;
		LabeledToken labeledToken = (LabeledToken) entity;
		seqOutput.setLabel(labeledToken.getOffset(),
				labelEncoding.put(labeledToken.getType()));
	}

	@Override
	public void clearEncodedEntities(ExampleInput input, ExampleOutput output) {
	}

}
