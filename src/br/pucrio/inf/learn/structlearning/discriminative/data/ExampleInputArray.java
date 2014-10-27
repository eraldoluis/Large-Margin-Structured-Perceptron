package br.pucrio.inf.learn.structlearning.discriminative.data;

import java.io.IOException;
import java.util.Collection;

public interface ExampleInputArray {

	ExampleInput get(int index);

	void put(ExampleInput input) throws IOException, DatasetException;
	void put(Collection<ExampleInput> inputs) throws IOException, DatasetException;
	void put(ExampleInputArray input) throws IOException, DatasetException;

	int getNumberExamples();

	void load(int[] index);
	void loadInOrder();
}
