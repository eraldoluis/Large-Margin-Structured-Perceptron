package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Collection;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.FeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

public class BisectionTemplate implements FeatureTemplate {

	/**
	 * Template index.
	 */
	private int index;

	/**
	 * Categorical basic features used in this template.
	 */
	private int[] categoricalFeatures;

	/**
	 * Numerical basic features used in this template.
	 */
	private int[] numericalFeatures;

	/**
	 * Temporary feature used to instantiate new features.
	 */
	private final Feature tempFeature;

	/**
	 * Create a template that combines the given features (categorical and
	 * numerical).
	 * 
	 * The given arrays of features indexes are not cloned by this constructor.
	 * Thus, the user must not modify them after creating this object.
	 * 
	 * @param index
	 * @param categoricalFeatures
	 * @param numericalFeatures
	 */
	public BisectionTemplate(int index, int[] categoricalFeatures,
			int[] numericalFeatures) {
		this.index = index;
		this.categoricalFeatures = categoricalFeatures;
		this.numericalFeatures = numericalFeatures;
		/*
		 * The feature object is used to encode derived features, i.e., to
		 * convert combined features (derived from templates) to codes. A unique
		 * derived feature is the combination of the values of its categorical
		 * features, which are given by the corresponding template. The values
		 * of numerical features are not used to encode a feature. The numerical
		 * values are used to scale (i.e., to multiply) the derived feature
		 * value when used by a model.
		 */
		this.tempFeature = new Feature(index,
				new int[categoricalFeatures.length]);
	}

	/**
	 * Create a template that combines the given features (categorical and
	 * numerical).
	 * 
	 * @param index
	 * @param categoricalFeatures
	 * @param numericalFeatures
	 */
	public BisectionTemplate(int index,
			Collection<Integer> categoricalFeaturesList,
			Collection<Integer> numericalFeaturesList) {
		this.index = index;
		this.categoricalFeatures = new int[categoricalFeaturesList.size()];
		int idx = 0;
		for (int ftr : categoricalFeaturesList)
			this.categoricalFeatures[idx++] = ftr;
		this.numericalFeatures = new int[numericalFeaturesList.size()];
		idx = 0;
		for (int ftr : numericalFeaturesList)
			this.numericalFeatures[idx++] = ftr;
		/*
		 * The feature object is used to encode derived features, i.e., to
		 * convert combined features (derived from templates) to codes. A unique
		 * derived feature is the combination of the values of its categorical
		 * features, which are given by the corresponding template. The values
		 * of numerical features are not used to encode a feature. The numerical
		 * values are used to scale (i.e., to multiply) the derived feature
		 * value when used by a model.
		 */
		this.tempFeature = new Feature(index,
				new int[categoricalFeatures.length]);
	}

	@Override
	public int getIndex() {
		return index;
	}

	@Override
	public int[] getFeatures() {
		return categoricalFeatures;
	}
	
	public int[] getNumericalFeatures() {
		return numericalFeatures;
	}

	public Feature getInstance(BisectionInput input, int paper1, int paper2) {
		int[] basicFeatures = input.getBasicCategoricalFeatures(paper1, paper2);
		if (basicFeatures == null)
			return null;
		return getInstance(basicFeatures);
	}

	public Feature newInstance(BisectionInput input, int paper1, int paper2) {
		int[] basicFeatures = input.getBasicCategoricalFeatures(paper1, paper2);
		if (basicFeatures == null)
			return null;
		return newInstance(basicFeatures);
	}

	public Feature getInstance(int[] edgeCategoricalFeatureValues) {
		int[] tmpValues = tempFeature.getValues();
		for (int idx = 0; idx < categoricalFeatures.length; ++idx)
			tmpValues[idx] = edgeCategoricalFeatureValues[categoricalFeatures[idx]];
		return tempFeature;
	}

	public Feature newInstance(int[] edgeCategoricalFeatureValues) {
		int[] newValues = new int[categoricalFeatures.length];
		for (int idx = 0; idx < categoricalFeatures.length; ++idx)
			newValues[idx] = edgeCategoricalFeatureValues[categoricalFeatures[idx]];
		return new Feature(index, newValues);
	}

	@Override
	public Feature getInstance(ExampleInput input, Object... params) {
		return getInstance((BisectionInput) input, (Integer) params[0],
				(Integer) params[1]);
	}

	@Override
	public Feature newInstance(ExampleInput input, Object... params) {
		return newInstance((BisectionInput) input, (Integer) params[0],
				(Integer) params[1]);
	}

}
