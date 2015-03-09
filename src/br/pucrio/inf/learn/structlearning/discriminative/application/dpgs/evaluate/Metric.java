package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.evaluate;

import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSOutput;

public interface Metric {
	void evaluate(int epoch, DPGSOutput[] corrects, DPGSOutput[] predicteds);
}
