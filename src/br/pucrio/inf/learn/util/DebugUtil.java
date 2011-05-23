package br.pucrio.inf.learn.util;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.FeatureEncoding;

/**
 * Utilitary methods to print debug information.
 * 
 * @author eraldo
 * 
 */
public class DebugUtil {

	public static boolean print = false;

	public static FeatureEncoding<String> featureEncoding = null;

	public static FeatureEncoding<String> stateEncoding = null;

	public static void printSequence(SequenceInput seqIn,
			SequenceOutput seqCorOut, SequenceOutput seqPreOut, double loss) {
		System.out.println("\nLoss: " + loss);
		System.out.print(" id=" + seqIn.getId() + "  ");
		for (int tkn = 0; tkn < seqIn.size(); ++tkn) {
			System.out.print(featureEncoding.getValueByCode(seqIn.getFeature(
					tkn, 0))
					+ "_"
					+ stateEncoding.getValueByCode(seqCorOut.getLabel(tkn))
					+ "_"
					+ stateEncoding.getValueByCode(seqPreOut.getLabel(tkn))
					+ "  ");
		}
		System.out.println();
	}

}
