package tagger.learning;

import java.io.PrintStream;

public class Verbose_res {
	public String L;
	public int nobjects, nanswers, nfullycorrect;

	public Verbose_res() {
		L = "";
		nobjects = 0;
		nanswers = 0;
		nfullycorrect = 0;
	}

	public Verbose_res(String label) {
		L = label;
		nobjects = 0;
		nanswers = 0;
		nfullycorrect = 0;
	}

	public Verbose_res(String _L, int _no, int _na) {
		L = _L;
		nobjects = _no;
		nanswers = _na;
		nfullycorrect = 0;
	}

	/**
	 * Calculate the precision corresponding to the performance values.
	 * 
	 * @return the precision.
	 */
	public double getPrecision() {
		if (nanswers == 0)
			return 0.0;
		return nfullycorrect / (double) nanswers;
	}

	/**
	 * Calculate the recall corresponding to the performance values.
	 * 
	 * @return the recall.
	 */
	public double getRecall() {
		if (nobjects == 0)
			return 0.0;
		return nfullycorrect / (double) nobjects;
	}

	/**
	 * Calculate the F-1 corresponding to the performance values.
	 * 
	 * @return the F-1.
	 */
	public double getF1() {
		double p = getPrecision();
		double r = getRecall();
		if (p + r == 0.0)
			return 0.0;
		return 2 * p * r / (p + r);
	}

	public void dump() {
		dump(System.err);
	}

	public void dump(PrintStream cout) {
		cout.println("L=" + L);
	}
}
