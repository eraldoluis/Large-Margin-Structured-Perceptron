package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;

/**
 * Feature template that conjoins a given set of basic features.
 * 
 * @author eraldo
 * 
 */
public class SimpleFeatureTemplate implements FeatureTemplate {

	/**
	 * Feature combination given by feature indexes.
	 */
	private int[] features;

	/**
	 * Create a template with the given feature combination.
	 * 
	 * @param features
	 */
	public SimpleFeatureTemplate(int[] features) {
		this.features = features;
	}

	@Override
	public Feature instantiate(DPInput input, int idxHead, int idxDep,
			int idxTemplate) {
		int[] edgeFeatures = input.getFeatureCodes(idxHead, idxDep);
		if (edgeFeatures.length > 0) {
			int[] values = new int[features.length];
			for (int idx = 0; idx < features.length; ++idx)
				values[idx] = edgeFeatures[features[idx]];
			return new Feature(idxTemplate, values);
		}

		return null;
	}

}
