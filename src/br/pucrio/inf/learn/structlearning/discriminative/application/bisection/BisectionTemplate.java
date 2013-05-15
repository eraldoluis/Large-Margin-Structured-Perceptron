package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.FeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

public class BisectionTemplate implements FeatureTemplate {

	/**
	 * Template index.
	 */
	private int index;

	/**
	 * Feature combination given by feature indexes.
	 */
	private int[] features;

	/**
	 * Temporary feature used to instantiate new features.
	 */
	private final Feature tempFeature;

	/**
	 * Create a template with the given feature combination.
	 * 
	 * @param features
	 */
	public BisectionTemplate(int index, int[] features) {
		this.index = index;
		this.features = features;
		this.tempFeature = new Feature(index, new int[features.length]);
	}

	@Override
	public int getIndex() {
		return index;
	}

	@Override
	public int[] getFeatures() {
		return features;
	}

	/**
	 * Return a temporary instance of this feature template for the given item.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDep
	 * @return
	 */
	public Feature getInstance(BisectionInput input, int item) {
		int[] basicFeatures = input.getBasicFeatures(item);
		if (basicFeatures == null)
			return null;
		return getInstance(basicFeatures);
	}

	/**
	 * Create a new instance of this feature template for the given edge. The
	 * returned instance can be used with no restrictions.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDep
	 * @return
	 */
	public Feature newInstance(BisectionInput input, int item) {
		int[] basicFeatures = input.getBasicFeatures(item);
		if (basicFeatures == null)
			return null;
		return newInstance(basicFeatures);
	}

	public Feature getInstance(int[] values) {
		int[] tmpValues = tempFeature.getValues();
		for (int idx = 0; idx < features.length; ++idx) {
			tmpValues[idx] = values[features[idx]];
		}
		tempFeature.setTemplateIndex(index);
		return tempFeature;
	}

	public Feature newInstance(int[] values) {
		int[] newValues = new int[features.length];
		for (int idx = 0; idx < features.length; ++idx)
			newValues[idx] = values[features[idx]];
		return new Feature(index, newValues);
	}

	@Override
	public Feature getInstance(ExampleInput input, Object... params) {
		return getInstance((BisectionInput) input, (Integer) params[0]);
	}

	@Override
	public Feature newInstance(ExampleInput input, Object... params) {
		return newInstance((BisectionInput) input, (Integer) params[0]);
	}

}
