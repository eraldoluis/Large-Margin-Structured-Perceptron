package br.pucrio.inf.learn.structlearning.application.sequence;

import br.pucrio.inf.learn.structlearning.data.Feature;

public interface SequenceFeatureTemplate {

	Feature instance(SequenceInput inputSequence, SequenceOutput outputSequence, int token);

}
