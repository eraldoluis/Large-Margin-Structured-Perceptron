package br.pucrio.inf.learn.structlearning.discriminative.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class SimpleExampleInputArray implements ExampleInputArray {
	private static final int INITIAL_CAPACITY = 50; 
	List<ExampleInput> list;
	
	
	public SimpleExampleInputArray() {
		list = new ArrayList<ExampleInput>(INITIAL_CAPACITY);
	}
	
	public SimpleExampleInputArray(int initialCapacity) {
		list = new ArrayList<ExampleInput>(initialCapacity);
	}
	
	@Override
	public ExampleInput get(int index) {
		// TODO Auto-generated method stub
		return list.get(index);
	}

	@Override
	public void put(ExampleInput input) throws IOException, DatasetException {
		// TODO Auto-generated method stub
		list.add(input);
	}

	@Override
	public int getNumberExamples() {
		// TODO Auto-generated method stub
		return list.size();
	}

	@Override
	public void load(int[] index) {
		
	}
	
	public void loadInOrder() {
		
	}
	
	public void close(){
		
	}

	@Override
	public void put(Collection<ExampleInput> inputs) throws IOException,
			DatasetException {
		list.addAll(inputs);
	}

	@Override
	public void put(ExampleInputArray input) throws IOException,
			DatasetException {
		input.loadInOrder();
		
		for (int i = 0; i < input.getNumberExamples(); i++) {
			put(input.get(i));
		}
	}


}
