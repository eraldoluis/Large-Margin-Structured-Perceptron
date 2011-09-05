package br.pucrio.inf.learn.structlearning.discriminative.evaluation;

/**
 * Generic precision and recall data and methods to calculate f1.
 * 
 * @author eraldo
 * 
 */
public class F1Measure {

	private String caption;

	private int numObjects;

	private int numPredicted;

	private int numCorrectlyPredicted;

	public F1Measure(String caption) {
		this.caption = caption;
		this.numObjects = 0;
		this.numPredicted = 0;
		this.numCorrectlyPredicted = 0;
	}

	public double getPrecision() {
		if (numPredicted == 0)
			return 0d;
		return ((double) numCorrectlyPredicted) / numPredicted;
	}

	public double getRecall() {
		if (numObjects == 0)
			return 0d;
		return ((double) numCorrectlyPredicted) / numObjects;
	}

	public double getF1() {
		double p = getPrecision();
		double r = getRecall();
		if (p + r == 0d)
			return 0d;
		return 2 * p * r / (p + r);
	}

	public String getCaption() {
		return caption;
	}

	public void setCaption(String caption) {
		this.caption = caption;
	}

	public int incNumObjects() {
		return ++numObjects;
	}

	public int incNumPredicted() {
		return ++numPredicted;
	}

	public int incNumCorrectlyPredicted() {
		return ++numCorrectlyPredicted;
	}

	/**
	 * Absolute number of correct objects. That is equal to the sum of true
	 * positives (correctly retrieved objects) and false negatives (missed
	 * objects).
	 * 
	 * @return
	 */
	public int getNumObjects() {
		return numObjects;
	}

	public void setNumObjects(int numObjects) {
		this.numObjects = numObjects;
	}

	/**
	 * Absolute number of retrieved objects (both correct and incorrect). That
	 * is equal to the sum of true positives (correctly retrieved objects) and
	 * false positives (incorrectly retrieved objects).
	 * 
	 * @return
	 */
	public int getNumRetrieved() {
		return numPredicted;
	}

	public void setNumPredicted(int numPredicted) {
		this.numPredicted = numPredicted;
	}

	/**
	 * Absolute number of correctly retrieved objects. That is equal to true
	 * positives.
	 * 
	 * @return
	 */
	public int getNumCorrectlyRetrieved() {
		return numCorrectlyPredicted;
	}

	public void setNumCorrectlyPredicted(int numCorrectlyPredicted) {
		this.numCorrectlyPredicted = numCorrectlyPredicted;
	}

}
