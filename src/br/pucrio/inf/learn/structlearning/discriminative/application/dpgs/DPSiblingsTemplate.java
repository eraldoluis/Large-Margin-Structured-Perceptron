package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.List;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;

public class DPSiblingsTemplate extends DPGSTemplate {

	/**
	 * Create a grandparent template with the given feature combination.
	 * 
	 * @param type
	 *            it must be 2 for left siblings factors and 3 for right
	 *            siblings factors.
	 * @param index
	 * @param featureIndexes
	 */
	public DPSiblingsTemplate(int type, int index, int[] featureIndexes) {
		this.index = index;
		this.featureIndexes = featureIndexes;
		int[] vals = new int[featureIndexes.length + 1];
		this.tempFeature = new Feature(index, vals);
		// Template type is 1 for grandparent templates.
		vals[featureIndexes.length] = type;
	}

	/**
	 * Instantiate siblings features for the grandparent factor of the given
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
	public void instantiateSiblingsDerivedFeatures(DPGSInput input,
			List<Integer> derivedFeatures, MapEncoding<Feature> encoding,
			int idxHead, int idxModifier, int idxGrandparent)
			throws CloneNotSupportedException {
		int[][] basicFeatures = input.getBasicSiblingsFeatures(idxHead,
				idxModifier, idxGrandparent);
		instantiateDerivedFeatures(basicFeatures, 0, derivedFeatures, encoding,
				tempFeature.getValues());
	}

}
