package br.pucrio.inf.learn.structlearning.task;

import br.pucrio.inf.learn.structlearning.data.FeatureVector;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

/**
 * Methods that should be implemented for each task so that one can use a
 * structural algorithm for this task.
 * 
 * @author eraldof
 * 
 */
public interface TaskAdapter {

	public FeatureVector extractFeatures(ExampleInput input,
			ExampleOutput output);

	public ExampleOutput inference(FeatureVector weight, ExampleInput input);

}
