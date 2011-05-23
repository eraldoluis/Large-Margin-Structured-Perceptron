package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.Arrays;
import java.util.Collection;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Sequence of labels for an input sequence.
 * 
 * @author eraldo
 * 
 */
public class ArraySequenceOutput implements SequenceOutput {

	/**
	 * Sequence of labels.
	 */
	private int[] labels;

	protected ArraySequenceOutput() {
	}

	public ArraySequenceOutput(int size) {
		labels = new int[size];
	}

	public ArraySequenceOutput(Iterable<Integer> labels, int size) {
		this(size);
		int idx = 0;
		for (int label : labels) {
			this.labels[idx] = label;
			++idx;
		}
	}

	public ArraySequenceOutput(Collection<Integer> labels) {
		this(labels, labels.size());
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

	@Override
	public ExampleOutput createNewObject() {
		return new ArraySequenceOutput(labels.length);
	}

	@Override
	public Object clone() throws CloneNotSupportedException {
		ArraySequenceOutput copy = new ArraySequenceOutput();
		copy.labels = labels.clone();
		return copy;
	}

	@Override
	public boolean equals(Object obj) {
		if (getClass() != obj.getClass())
			return false;
		return Arrays.equals(labels, ((ArraySequenceOutput) obj).labels);
	}

}
