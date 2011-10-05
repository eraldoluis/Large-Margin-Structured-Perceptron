package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Represent a PQ output structure for a document, i.e., a vector of people, one
 * for each quotation.
 * 
 * @author eraldo
 * 
 */
public class PQOutput implements ExampleOutput {

	/**
	 * Person index for each quotation.
	 */
	private int[] personByQuotation;

	public PQOutput(int numberOfQuotations) {
		personByQuotation = new int[numberOfQuotations];
	}

	@Override
	public ExampleOutput createNewObject() {
		// TODO Auto-generated method stub
		return null;
	}

}
