package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;


/**
 * Model parameter that supports a voted perceptron implementation.
 * 
 * @author eraldof
 * 
 */
public final class AveragedParameter implements Comparable<AveragedParameter>,
		Cloneable {

	/**
	 * The current (non-averaged) weight. This value must be used by the
	 * inference algorithm through the Perceptron execution.
	 */
	private double weight;

	/**
	 * Update realized within the current iteration. This must be summed to the
	 * <code>sum</code> value at the end of each iteration.
	 */
	private double update;

	/**
	 * The current sum of the values assumed by this weight in all previous
	 * iterations.
	 */
	private double sum;

	/**
	 * Last iteration when this weight was summed (<code>update</code> value was
	 * summed into the <code>sum</code> value).
	 */
	private int lastSummedIteration;

	/**
	 * Set the value of this weight.
	 * 
	 * @param value
	 */
	public void set(double value) {
		weight = value;
		sum = 0d;
		update = 0d;
	}

	/**
	 * Add the given value <code>val</code> to this weight. In fact, this value
	 * is added to the <code>update</code> before being incorporated in the
	 * weight itself.
	 * 
	 * @param value
	 */
	public void update(double value) {
		update += value;
	}

	/**
	 * Return the current value of this weight.
	 * 
	 * @return
	 */
	public double get() {
		return weight;
	}

	/**
	 * Account the last updates in its weight and in its summed (for later
	 * averaging) value.
	 * 
	 * @param iteration
	 */
	public void sum(int iteration) {
		sum += weight * (iteration - lastSummedIteration) + update;
		weight += update;
		update = 0d;
		lastSummedIteration = iteration;
	}

	/**
	 * Average this weight.
	 * 
	 * @param numberOfIterations
	 *            total number of iterations of the training algorithm.
	 */
	public void average(int numberOfIterations) {
		// Account any residual value.
		sum(numberOfIterations - 1);
		// Average.
		weight = sum / numberOfIterations;
		// Keep track that this weight was already averaged.
		sum = Double.NEGATIVE_INFINITY;
	}

	@Override
	public int compareTo(AveragedParameter other) {
		if (this == other)
			return 0;
		int idThis = System.identityHashCode(this);
		int idOther = System.identityHashCode(other);
		if (idThis < idOther)
			return -1;
		return 1;
	}

	@Override
	public AveragedParameter clone() throws CloneNotSupportedException {
		return (AveragedParameter) super.clone();
	}

}
