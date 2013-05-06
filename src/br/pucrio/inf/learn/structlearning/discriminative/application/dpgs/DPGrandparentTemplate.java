package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.List;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;

/**
 * Feature template that conjoins a given set of basic features.
 * 
 * @author eraldo
 * 
 */
public class DPGrandparentTemplate extends DPGSTemplate {

	/**
	 * Create a grandparent template with the given feature combination.
	 * 
	 * @param index
	 * @param featureIndexes
	 */
	public DPGrandparentTemplate(int index, int[] featureIndexes) {
		this.index = index;
		this.featureIndexes = featureIndexes;
		int[] vals = new int[featureIndexes.length + 1];
		this.tempFeature = new Feature(index, vals);
		// Template type is 1 for grandparent templates.
		vals[featureIndexes.length] = 1;
	}

	/**
	 * Instantiate grandparent features for the grandparent factor of the given
	 * input.
	 * 
	 * @param input
	 * @param derivedFeatures
	 * @param encoding
	 * @param idxHead
	 * @param idxModifier
	 * @param idxGrandparent
	 * @throws CloneNotSupportedException
	 */
	public void instantiateGrandparentDerivedFeatures(DPGSInput input,
			List<Integer> derivedFeatures, MapEncoding<Feature> encoding,
			int idxHead, int idxModifier, int idxGrandparent)
			throws CloneNotSupportedException {
		int[][] basicFeatures = input.getBasicGrandparentFeatures(idxHead,
				idxModifier, idxGrandparent);
		instantiateDerivedFeatures(basicFeatures, 0, derivedFeatures, encoding);
	}
}
