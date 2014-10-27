package br.pucrio.inf.learn.structlearning.discriminative.algorithm.passiveagressive;

import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

public class PassiveAgressive implements OnlineStructuredAlgorithm{
	
	
	
	@Override
	public void train(ExampleInput[] inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInput[] inputsB,
			ExampleOutput[] outputsB) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Model getModel() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setPartiallyAnnotatedExamples(boolean value) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setListener(TrainingListener listener) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setSeed(long seed) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double train(ExampleInput input, ExampleOutput output,
			ExampleOutput predicted) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setLearningRate(double rate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getIteration() {
		// TODO Auto-generated method stub
		return 0;
	}

}
