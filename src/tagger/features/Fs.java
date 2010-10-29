package tagger.features;

import java.util.Random;

/**
 * Weight of a unique model parameter, i.e., associated with a unique feature.
 * Includes methods and fields to help the averaged perceptron algorithm.
 * 
 */
public class Fs {

	/*
	 * TODO: jordi find a better solution to setup the random generator.
	 * 
	 * @eraldo: I suggest a unique class with a static random generator object.
	 * Hence, all SST code can use the same generator. This class can be in the
	 * tagger.utils package, for example.
	 */
	public static Random generator = new Random();

	/**
	 * Parse the given string that must contains a serialized parameter.
	 * 
	 * @param readLine
	 *            a string containing a serialized parameter.
	 * 
	 * @return a new parameter object.
	 */
	public static Fs parseFs(String readLine) {
		String buff[] = readLine.split("[ \t]");
		return new Fs(Double.parseDouble(buff[0]), Double.parseDouble(buff[1]),
				Integer.parseInt(buff[2]), Double.parseDouble(buff[3]));
	}

	/**
	 * Current weight (non averaged) for the parameter.
	 */
	public double alpha;

	/**
	 * Numerator of the averaged weight for the parameter.
	 */
	public double avg_num;

	/**
	 * Iteration number of the last update.
	 */
	public int lst_upd;

	/**
	 * Accumulated update value.
	 * 
	 * This value is public, so it may be updated wherever. However, at some
	 * point, it must be added to <code>alpha</code> by calling the
	 * <code>update</code> method.
	 */
	public double upd_val;

	/**
	 * Default constructor.
	 */
	public Fs() {
		alpha = 0;
		avg_num = 0;
		lst_upd = 0;
		upd_val = 0;
	}

	/**
	 * Full constructor.
	 * 
	 * @param alpha
	 * @param avg_num
	 * @param lst_upd
	 * @param upd_val
	 */
	public Fs(double alpha, double avg_num, int lst_upd, double upd_val) {
		this.alpha = alpha;
		this.avg_num = avg_num;
		this.lst_upd = lst_upd;
		this.upd_val = upd_val;
	}

	/**
	 * Return the averaged value of the parameter.
	 * 
	 * @param it
	 *            current iteration number of the perceptron algorithm.
	 * @param use_avg
	 *            if <code>true</code> then use the averaged parameter.
	 * 
	 * @return the averaged weight of the parameter.
	 */
	public double avg_val(int it, boolean use_avg) {
		if (!use_avg)
			return alpha;
		return (avg_num + (alpha * (it - lst_upd))) / (double) it;
	}

	/**
	 * Update the <code>alpha</code> and <code>avg_num</code> values using the
	 * current <code>upd_val</code> value.
	 * 
	 * @param it
	 *            the current perceptron iteration.
	 * @param doupdate
	 *            if <code>true</code> then update the internal accumulators and
	 *            clear the <code>upd_val</code>, if <code>false</code> then
	 *            just clear the <code>upd_val</code> value.
	 */
	public void update(int it, boolean doupdate) {
		if (upd_val != 0 && doupdate) {
			avg_num += alpha * (it - lst_upd);
			lst_upd = it;
			alpha += upd_val;
		}
		upd_val = 0;
	}

	public void equate(Fs c) {
		alpha = c.alpha;
		avg_num = c.avg_num;
		lst_upd = c.lst_upd;
		upd_val = c.upd_val;
	}

	public void rand_alpha() {
		// double r = rand()/double(RAND_MAX), p = rand()/double(RAND_MAX);
		double r = (double) generator.nextDouble(); // @AJAB
													// ??generator.nextGaussian();
		double p = (double) generator.nextDouble();
		if (p > 0.5)
			alpha = r;
		else
			alpha = r * -1;
	}

	public String dump() {
		return "alpha " + String.format("%.2f", alpha) + " avg "
				+ String.format("%.2f", avg_num) + " lst " + lst_upd + " upd "
				+ String.format("%.2f", upd_val);
	}

	public String toString() {
		return dump();
	}
}
