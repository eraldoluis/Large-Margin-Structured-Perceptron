package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.util.Arrays;

import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Represents a feature template instantiation.
 * 
 * @author eraldo
 * 
 */
public class Feature {

	/**
	 * Template index that generated this feature.
	 */
	private int template;

	/**
	 * Values for each basic feature in this combined feature.
	 */
	private int[] values;

	/**
	 * Instantiate a new feature.
	 * 
	 * @param template
	 * @param values
	 */
	public Feature(int template, int[] values) {
		this.template = template;
		this.values = values;
	}

	@Override
	public int hashCode() {
		return HashCodeUtil.hash(template, values);
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Feature))
			return false;
		Feature f = (Feature) o;
		return template == f.template && Arrays.equals(values, f.values);
	}

}
