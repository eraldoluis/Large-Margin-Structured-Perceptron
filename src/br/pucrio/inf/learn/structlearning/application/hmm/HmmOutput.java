package br.pucrio.inf.learn.structlearning.application.hmm;

import java.util.Vector;

import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

public class HmmOutput implements ExampleOutput {

	private Vector<Integer> labels;

	public int size() {
		return labels.size();
	}

	public int getLabel(int token) {
		return labels.get(token);
	}

}
