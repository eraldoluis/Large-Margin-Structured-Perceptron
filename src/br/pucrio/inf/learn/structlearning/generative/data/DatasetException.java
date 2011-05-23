package br.pucrio.inf.learn.structlearning.generative.data;

public class DatasetException extends Exception {
	/**
	 * Serial version UID.
	 */
	private static final long serialVersionUID = 4965845719147927074L;

	public DatasetException(String message) {
		super(message);
	}

	public DatasetException(String message, Throwable thw) {
		super(message, thw);
	}
}
