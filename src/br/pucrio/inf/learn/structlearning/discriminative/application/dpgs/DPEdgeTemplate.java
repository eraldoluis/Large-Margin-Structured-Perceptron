package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.util.List;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;

public class DPEdgeTemplate extends DPGSTemplate {

	/**
	 * Create an edge template with the given feature combination.
	 * 
	 * @param index
	 * @param featureIndexes
	 */
	public DPEdgeTemplate(int index, int[] featureIndexes) {
		this.index = index;
		this.featureIndexes = featureIndexes;
		int[] vals = new int[featureIndexes.length + 1];
		this.tempFeature = new Feature(index, vals);
		// Template type is 0 for edge templates.
		vals[featureIndexes.length] = 0;
	}

	/**
	 * Instantiate edge features for the edge factor of the given input.
	 * 
	 * @param input
	 * @param derivedFeatures
	 * @param encoding
	 * @param idxHead
	 * @param idxModifier
	 * @throws CloneNotSupportedException
	 */
	public void instantiateEdgeDerivedFeatures(DPGSInput input,
			List<Integer> derivedFeatures, MapEncoding<Feature> encoding,
			int idxHead, int idxModifier) throws CloneNotSupportedException {
		int[][] basicFeatures = input
				.getBasicEdgeFeatures(idxHead, idxModifier);
		instantiateDerivedFeatures(basicFeatures, 0, derivedFeatures, encoding);
	}

}
