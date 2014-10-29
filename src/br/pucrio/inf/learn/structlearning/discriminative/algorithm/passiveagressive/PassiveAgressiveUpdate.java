package br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

public interface PassiveAgressiveUpdate {
	
	ExampleOutput getExampleOutput(ExampleOutput correct, ExampleOutput predict);
}
