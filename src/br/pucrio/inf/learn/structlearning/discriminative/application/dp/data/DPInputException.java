package br.pucrio.inf.learn.structlearning.discriminative.application.dp.data;

/**
 * Exception within <code>DPInput</code> code.
 * 
 * @author eraldo
 * 
 */
public class DPInputException extends Exception {

	/**
	 * Auto-generated serial ID.
	 */
	private static final long serialVersionUID = 2350018856859717707L;

	public DPInputException() {
		super("Dependency Parsing input structure error");
	}

	public DPInputException(String message) {
		super(message);
	}

	public DPInputException(Throwable cause) {
		super(cause);
	}

	public DPInputException(String message, Throwable cause) {
		super(message, cause);
	}

}
