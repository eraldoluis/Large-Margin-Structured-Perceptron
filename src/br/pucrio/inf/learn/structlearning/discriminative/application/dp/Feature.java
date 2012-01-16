package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.util.Arrays;

import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Represent a feature template instantiation. Store the template index and the
 * value of each feature in the template.
 * 
 * @author eraldo
 * 
 */
public class Feature implements Cloneable {

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

	/**
	 * Return the template index of this feature.
	 * 
	 * @return
	 */
	public int getTemplate() {
		return template;
	}

	/**
	 * Return the values for each basic feature within this composed feature.
	 * 
	 * @return
	 */
	public int[] getValues() {
		return values;
	}

	/**
	 * Set the internal vector of values.
	 * 
	 * @param values
	 */
	public void setValues(int[] values) {
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

	/**
	 * Clone this instance.
	 */
	public Feature clone() throws CloneNotSupportedException {
		return new Feature(template, values.clone());
	}

}
