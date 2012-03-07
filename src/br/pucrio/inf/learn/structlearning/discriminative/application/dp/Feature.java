package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.util.Arrays;

import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Represent a feature templateIndex instantiation. Store the templateIndex
 * index and the value of each feature in the templateIndex.
 * 
 * @author eraldo
 * 
 */
public class Feature implements Cloneable {

	/**
	 * Template index that generated this feature.
	 */
	private int templateIndex;

	/**
	 * Values for each basic feature in this combined feature.
	 */
	private int[] values;

	/**
	 * Instantiate a new feature.
	 * 
	 * @param templateIndex
	 * @param values
	 */
	public Feature(int templateIndex, int[] values) {
		this.templateIndex = templateIndex;
		this.values = values;
	}

	/**
	 * Return the templateIndex index of this feature.
	 * 
	 * @return
	 */
	public int getTemplateIndex() {
		return templateIndex;
	}

	/**
	 * Set this feature template index.
	 * 
	 * @param index
	 */
	public void setTemplateIndex(int index) {
		templateIndex = index;
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
		return HashCodeUtil.hash(templateIndex, values);
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Feature))
			return false;
		Feature f = (Feature) o;
		return templateIndex == f.templateIndex
				&& Arrays.equals(values, f.values);
	}

	/**
	 * Clone this instance.
	 */
	public Feature clone() throws CloneNotSupportedException {
		return new Feature(templateIndex, values.clone());
	}

}
