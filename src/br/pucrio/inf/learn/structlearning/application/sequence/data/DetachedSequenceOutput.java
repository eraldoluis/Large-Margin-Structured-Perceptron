package br.pucrio.inf.learn.structlearning.application.sequence.data;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;

public class DetachedSequenceOutput implements SequenceOutput {

	private int[] labels;

	public DetachedSequenceOutput(int size) {
		labels = new int[size];
	}

	public DetachedSequenceOutput(int[] copy, boolean deep) {
		if (deep)
			labels = copy.clone();
		else
			labels = copy;
	}

	@Override
	public int size() {
		return labels.length;
	}

	@Override
	public int getLabel(int token) {
		return labels[token];
	}

	@Override
	public void setLabel(int token, int label) {
		labels[token] = label;
	}

}
