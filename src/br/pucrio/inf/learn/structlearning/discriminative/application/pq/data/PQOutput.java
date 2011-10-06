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
	 * Person index.
	 */
	private int person;

	
	public PQOutput() {
		this.person = -1;
	}
	
	public PQOutput(int indexPerson) {
		this.person = indexPerson;
	}

	@Override
	public ExampleOutput createNewObject() {
		return new PQOutput();
	}

	public void setPerson(int indexPerson) {
		this.person = indexPerson;
	}
	
	public int getPerson() {
		return this.person;
	}


}
