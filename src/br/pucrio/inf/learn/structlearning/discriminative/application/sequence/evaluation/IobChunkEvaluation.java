package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation;

import java.util.Collection;

import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.EntityF1Evaluation;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.TypedEntity;

/**
 * Provide methods to evaluate precision, recall and F1 values of sequences that
 * contain chunks codified with IOB tagging style.
 * 
 * @author eraldof
 * 
 */
public class IobChunkEvaluation extends EntityF1Evaluation {

	/**
	 * Label that codifies no chunk information.
	 */
	private String nullLabel;

	/**
	 * Code of the null label.
	 */
	private int nullLabelCode;

	/**
	 * The encoding for state labels.
	 */
	private FeatureEncoding<String> stateEncoding;

	/**
	 * If <code>true</code>, always use the B- tag in the first token of an
	 * entity, even when there is no entity of the same type immediately before
	 * this token.
	 */
	private boolean alwaysUseBTag;

	/**
	 * Create an evaluation object. The user must provide the state-label
	 * encoding and the null label.
	 * 
	 * @param stateEncoding
	 * @param nullLabel
	 */
	public IobChunkEvaluation(FeatureEncoding<String> stateEncoding,
			String nullLabel) {
		this.stateEncoding = stateEncoding;
		this.nullLabel = nullLabel;
		this.nullLabelCode = stateEncoding.put(nullLabel);
	}

	/**
	 * Extract the chunks within the given output sequence.
	 * 
	 * @param input
	 *            the input sequence. Usually, this value is not necessary to
	 *            extract the chunks.
	 * @param output
	 *            the output sequence.
	 * @param chunks
	 *            the extracted chunks will be added to this collection.
	 */
	public void extractEntities(SequenceInput input, SequenceOutput output,
			Collection<TypedEntity> chunks) {

		int idxTknBegin = 0;
		String curType = nullLabel;

		int lenExample = output.size();
		for (int idxTkn = 0; idxTkn < lenExample; ++idxTkn) {
			String tag = stateEncoding.getValueByCode(output.getLabel(idxTkn));
			if (tag == null)
				tag = nullLabel;

			String beg = nullLabel;
			String type = nullLabel;

			String[] strs = tag.split("-", 2);
			beg = strs[0];
			if (strs.length > 1)
				type = strs[1];
			else
				type = beg;

			// Find the beginning of an entity (maybe an "O entity").
			if (!type.equals(curType) || beg.equals("B")) {
				// If the previous entity is a valid one (not "O entity").
				if (!curType.equals(nullLabel))
					chunks.add(new TypedChunk(idxTknBegin, idxTkn - 1, curType));

				// Restart the current entity.
				idxTknBegin = idxTkn;
				curType = type;
			}
		}

		// If the last entity ends at the last token of the sentence.
		if (!curType.equals(nullLabel))
			chunks.add(new TypedChunk(idxTknBegin, lenExample - 1, curType));
	}

	/**
	 * Tag the output sequence with the given chunk.
	 * 
	 * @param inputSeq
	 * @param outputSeq
	 * @param entity
	 */
	private void tagEntity(SequenceInput inputSeq, SequenceOutput outputSeq,
			TypedEntity entity) {
		// Cast.
		TypedChunk chunk = (TypedChunk) entity;

		// Label prefixes.
		String beg, in;
		if (chunk.type.length() == 0) {
			beg = "B";
			in = "I";
		} else {
			beg = "B-";
			in = "I-";
		}

		// Check the first token tag (B- or I-).
		String firstPrefix = beg;
		if (!alwaysUseBTag) {
			if (chunk.tokenBeg == 0)
				firstPrefix = in;
			else {
				String prevTag = stateEncoding.getValueByCode(outputSeq
						.getLabel(chunk.tokenBeg - 1));
				if (!prevTag.equals(beg + chunk.type)
						&& !prevTag.equals(in + chunk.type))
					firstPrefix = in;
			}
		}

		// Tag the output sequence with the given entity.
		outputSeq.setLabel(chunk.tokenBeg,
				stateEncoding.put(firstPrefix + chunk.type));
		for (int tkn = chunk.tokenBeg + 1; tkn <= chunk.tokenEnd; ++tkn)
			outputSeq.setLabel(tkn, stateEncoding.put(in + chunk.type));
	}

	@Override
	public void decodeEntities(ExampleInput input, ExampleOutput output,
			Collection<TypedEntity> entities) {
		extractEntities((SequenceInput) input, (SequenceOutput) output,
				entities);
	}

	@Override
	public void encodeEntity(ExampleInput input, ExampleOutput output,
			TypedEntity entity) {
		tagEntity((SequenceInput) input, (SequenceOutput) output, entity);
	}

	@Override
	public void clearEncodedEntities(ExampleInput input, ExampleOutput output) {
		SequenceOutput seqOutput = (SequenceOutput) output;
		for (int tkn = 0; tkn < seqOutput.size(); ++tkn)
			seqOutput.setLabel(tkn, nullLabelCode);
	}
}
