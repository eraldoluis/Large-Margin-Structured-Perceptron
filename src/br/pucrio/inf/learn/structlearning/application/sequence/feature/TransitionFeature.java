package br.pucrio.inf.learn.structlearning.application.sequence.feature;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.Feature;
import br.pucrio.inf.learn.util.HashCodeUtil;

public class TransitionFeature implements Feature {

	private static final int OUT_OF_SEQUENCE = Integer.MIN_VALUE;

	private int[] labelSequence;

	public TransitionFeature(SequenceInput inputSequence,
			SequenceOutput outputSequence, int centralToken, int order) {
		instance(inputSequence, outputSequence, centralToken, order);
	}

	public void instance(SequenceInput inputSequence,
			SequenceOutput outputSequence, int centralToken, int order) {
		labelSequence = new int[order + 1];

		// First valid token (first token can be out of the sequence, i.e., less
		// than zero).
		int beg = Math.max(0, centralToken - order);

		// Fill the label sequence.
		for (int token = beg; token <= centralToken; ++token)
			labelSequence[token - centralToken + order] = outputSequence
					.getLabel(token);

		// Fill out-of-sequence labels with a special (dummy) value.
		for (int token = centralToken - order; token < 0; ++token)
			labelSequence[token - centralToken + order] = OUT_OF_SEQUENCE;
	}

	@Override
	public int compareTo(Feature other) {
		if (getClass() != other.getClass())
			return getClass().toString().compareTo(other.getClass().toString());
		return compareTo((TransitionFeature) other);
	}

	public int compareTo(TransitionFeature other) {
		if (labelSequence.length < other.labelSequence.length)
			return -1;
		if (labelSequence.length > other.labelSequence.length)
			return 1;
		for (int offset = 0; offset < labelSequence.length; ++offset) {
			if (labelSequence[offset] < other.labelSequence[offset])
				return -1;
			if (labelSequence[offset] > other.labelSequence[offset])
				return 1;
		}
		return 0;
	}

	@Override
	public boolean equals(Object obj) {
		if (getClass() != obj.getClass())
			return false;
		return compareTo((TransitionFeature) obj) == 0;
	}

	@Override
	public int hashCode() {
		return HashCodeUtil.hash(labelSequence);
	}

}
