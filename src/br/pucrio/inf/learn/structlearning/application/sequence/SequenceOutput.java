package br.pucrio.inf.learn.structlearning.application.sequence;

import java.util.Collection;

import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

/**
 * Sequence of labels for an input sequence.
 * 
 * @author eraldo
 * 
 */
public class SequenceOutput implements ExampleOutput {

	/**
	 * Sequence of labels.
	 */
	private int[] labels;

	public SequenceOutput(int size) {
		labels = new int[size];
	}

	public SequenceOutput(Iterable<Integer> labels, int size) {
		this(size);
		int idx = 0;
		for (int label : labels) {
			this.labels[idx] = label;
			++idx;
		}
	}

	public SequenceOutput(Collection<Integer> labels) {
		this(labels, labels.size());
	}

	/**
	 * Return the number of tokens in this sequence.
	 * 
	 * @return
	 */
	public int size() {
		return labels.length;
	}

	/**
	 * Return the label of the given token.
	 * 
	 * @param token
	 * @return
	 */
	public int getLabel(int token) {
		return labels[token];
	}

	/**
	 * Set the label for the given token.
	 * 
	 * @param token
	 * @param label
	 */
	public void setLabel(int token, int label) {
		labels[token] = label;
	}

	@Override
	public ExampleOutput createNewObject() {
		return new SequenceOutput(labels.length);
	}

}
