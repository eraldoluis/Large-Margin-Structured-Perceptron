package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for person-quotation. A vector of people, one for each
 * quotation.
 * 
 * @author eraldo
 * 
 */
public class PQOutput implements ExampleOutput {
	/*
	 * Person index for each quotation.
	 */
	private int[] personByQuotation;

	public PQOutput(int numberOfQuotations) {
		personByQuotation = new int[numberOfQuotations];
	}

	@Override
	public ExampleOutput createNewObject() {
		return new PQOutput(personByQuotation.length);
	}

	/**
	 * Associate the given person with the given quotation.
	 * 
	 * @param indexQuotation
	 * @param indexPerson
	 */
	public void setPerson(int indexQuotation, int indexPerson) {
		personByQuotation[indexQuotation] = indexPerson;
	}

}
