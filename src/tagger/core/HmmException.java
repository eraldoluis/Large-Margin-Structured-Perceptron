package tagger.core;

public class HmmException extends Exception {

	/**
	 * Serial version UID.
	 */
	private static final long serialVersionUID = -239783673213241654L;

	public HmmException() {
	}

	public HmmException(String message) {
		super(message);
	}

	public HmmException(Throwable nested) {
		super(nested);
	}

	public HmmException(String message, Throwable nested) {
		super(message, nested);
	}

}
