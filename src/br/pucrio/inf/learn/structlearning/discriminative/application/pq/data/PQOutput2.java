package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import java.util.Arrays;
import java.util.Collection;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for person-quotation. A vector of people, one for each
 * quotation.
 * 
 * @author eraldo
 * 
 */
public class PQOutput2 implements ExampleOutput {
	/*
	 * Author indexes.
	 */
	private int[] authors;

	
	
	public PQOutput2(int size) {
		this.authors = new int[size];
	}
	
	public PQOutput2(Iterable<Integer> authors, int size) {
		this(size);
		int idx = 0;
		for (int author : authors) {
			this.authors[idx] = author;
			++idx;
		}
	}
	
	public PQOutput2(Collection<Integer> authors) {
		this(authors, authors.size());
	}

	public int size() {
		return authors.length;
	}

	public int getAuthor(int token) {
		return authors[token];
	}

	public void setAuthor(int token, int author) {
		authors[token] = author;
	}

	public ExampleOutput createNewObject() {
		return new PQOutput2(authors.length);
	}

	public boolean equals(Object obj) {
		if (getClass() != obj.getClass())
			return false;
		return Arrays.equals(authors, ((PQOutput2) obj).authors);
	}
	
	@Override
	public double getFeatureVectorLengthSquared(ExampleInput input, ExampleOutput other) {
		throw new NotImplementedException();
	}
}
