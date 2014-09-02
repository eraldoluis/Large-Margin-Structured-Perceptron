package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.Serializable;
import java.util.List;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;

/**
 * Feature template that conjoins a given set of basic features.
 * 
 * @author eraldo
 * 
 */
public class DPGSTemplate implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 100L;

	/**
	 * Template index.
	 */
	protected int index;

	/**
	 * Feature combination given by feature indexes.
	 */
	protected int[] featureIndexes;

	/**
	 * Temporary feature to avoid unecessary instantiation of existent features.
	 */
	protected Feature tempFeature;

	public int getIndex() {
		return index;
	}

	public int[] getFeatures() {
		return featureIndexes;
	}

	/**
	 * Instantiate derived features based on this template and the given array
	 * of basic features. New derived features are encoded by the given encoding
	 * and the resulting code is inserted in the <code>derivedFeatures</code>
	 * list.
	 * 
	 * @param basicFeatures
	 * @param idxFtrInTemplate
	 * @param derivedFeatures
	 * @param encoding
	 * @param currentValues
	 * @throws CloneNotSupportedException
	 */
	protected void instantiateDerivedFeatures(int[][] basicFeatures,
			int idxFtrInTemplate, List<Integer> derivedFeatures,
			MapEncoding<Feature> encoding) throws CloneNotSupportedException {
		int[] ftrs = basicFeatures[featureIndexes[idxFtrInTemplate]];
		int numFtrs = ftrs.length;

		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr) {
			int ftrVal = ftrs[idxFtr];
			tempFeature.setValue(idxFtrInTemplate, ftrVal);
			if (idxFtrInTemplate == featureIndexes.length - 1) {
				int code = encoding.getCodeByValue(tempFeature);
				if (code == FeatureEncoding.UNSEEN_VALUE_CODE)
					// Unseen feature. Create a new instance.
					code = encoding.put(tempFeature.clone());
				derivedFeatures.add(code);
			} else {
				instantiateDerivedFeatures(basicFeatures, idxFtrInTemplate + 1,
						derivedFeatures, encoding);
			}
		}
	}
}
