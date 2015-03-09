package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.evaluate;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSOutput;

public class PrecisionRecallF1  implements Metric {
	
	private final static Log LOG = LogFactory.getLog(PrecisionRecallF1.class);

	@Override
	public void evaluate(int epoch, DPGSOutput[] corrects,
			DPGSOutput[] predicteds) {

		DPGSOutput correct;
		DPGSOutput predict;
		int tpG = 0; // TRUE POSITIVE GRANDPARENT
		int fpG = 0; // FALSE POSITIVE GRANDPARENT
		int fnG = 0; // FALSE NEGATIVE GRANDPARENT

		int tpMod = 0; // TRUE POSITIVE MODIFIER
		int fpMod = 0; // FALSE POSITIVE MODIFIER
		int fnMod = 0; // FALSE NEGATIVE MODIFIER

		for (int i = 0; i < predicteds.length; i++) {
			correct = corrects[i];
			predict = predicteds[i];

			for (int idxHead = 0; idxHead < predict.size(); idxHead++) {
				if (correct.getHead(idxHead) == predict
						.getGrandparent(idxHead)) {
					tpG++;
				} else {
					fnG++;
					fpG++;
				}

				for (int idxModifier = 0; idxModifier < predict.size(); idxModifier++) {
					if (correct.getHead(idxModifier) == idxHead
							&& predict.isModifier(idxHead, idxModifier))
						tpMod++;
					else if (correct.getHead(idxModifier) == idxHead)
						fnMod++;
					else if (predict.isModifier(idxHead, idxModifier))
						fpMod++;
				}
			}
		}

		LOG.info("Evaluation after epoch " + epoch + ":");

		double beta = 1.0d;

		double precisionG = calculatePrecision(tpG, fpG);
		double recallG = calculateRecall(tpG, fnG);
		double fMeasureG = calculateFMeasure(beta, precisionG, recallG);

		System.out
				.println(String
						.format("\tPrecision, Recall e FMeasure(beta=%.2f) of Grandparent: %.2f%% / %.2f%% / %.2f%%",
								beta, precisionG * 100d, recallG * 100d,
								fMeasureG * 100d));

		double precisionMod = calculatePrecision(tpMod, fpMod);
		double recallMod = calculateRecall(tpMod, fnMod);
		double fMeasureMod = calculateFMeasure(beta, precisionMod, recallMod);

		System.out
				.println(String
						.format("\tPrecision, recall e fMeasure(beta=%.2f) of Modifiers: %.2f%% / %.2f%% / %.2f%%",
								beta, precisionMod * 100d,
								recallMod * 100d, fMeasureMod * 100d));

		int tp = tpMod + tpG;
		int fn = fnG + fnMod;
		int fp = fpG + fpMod;
		double precision = calculatePrecision(tp, fp);
		double recall = calculateRecall(tp, fn);
		double fMeasure = calculateFMeasure(beta, precision, recall);

		System.out
				.println(String
						.format("\tPrecision, recall e fMeasure(beta=%.2f) of Grandparent and Modifiers: %.2f%% / %.2f%% / %.2f%%",
								beta, precision * 100d, recall * 100d,
								fMeasure * 100d));

	}
	private double calculateRecall(int tp, int fn) {
		return tp / (double) (tp + fn);
	}
	private double calculatePrecision(int tp, int fp) {
		return tp / (double) (tp + fp);
	}

	private double calculateFMeasure(double beta, double precisionG,
			double recallG) {
		return ((beta * beta + 1) * precisionG * recallG)
				/ ((beta * beta * precisionG) + recallG);
	}

}
