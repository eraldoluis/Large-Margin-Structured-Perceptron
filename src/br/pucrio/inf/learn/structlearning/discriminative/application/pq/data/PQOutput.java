package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for person-quotation.
 * 
 * @author eraldo
 * 
 */
public class PQOutput implements ExampleOutput {

	/**
	 * Array with the index of the person associated with each quotation.
	 */
	private int[] person;

	public PQOutput(int numberOfQuotations) {
		person = new int[numberOfQuotations];
	}

	@Override
	public PQOutput createNewObject() {
		return new PQOutput(person.length);
	}

	/**
	 * Associate the given person with the given quotation.
	 * 
	 * @param indexQuotation
	 * @param indexPerson
	 */
	public void setPerson(int indexQuotation, int indexPerson) {
		person[indexQuotation] = indexPerson;
	}

}
