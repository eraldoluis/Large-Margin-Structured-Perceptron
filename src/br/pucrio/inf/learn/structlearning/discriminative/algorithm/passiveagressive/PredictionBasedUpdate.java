package br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

public class PredictionBasedUpdate implements PassiveAgressiveUpdate {

	@Override
	public ExampleOutput getExampleOutput(ExampleOutput correct, ExampleOutput predict) {
		return predict;
	}

}
