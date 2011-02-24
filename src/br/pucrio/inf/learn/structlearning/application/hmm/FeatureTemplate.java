package br.pucrio.inf.learn.structlearning.application.hmm;

import br.pucrio.inf.learn.structlearning.data.Feature;

public interface FeatureTemplate {

	Feature extractFeature(HmmInput inputSequence, HmmOutput outputSequence, int token) {
		
	}

}
