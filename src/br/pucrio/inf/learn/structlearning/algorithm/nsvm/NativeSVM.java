package br.pucrio.inf.learn.structlearning.algorithm.nsvm;

import br.pucrio.inf.learn.structlearning.algorithm.StructuredAlgorithm;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;

/**
 * Stub for Joachims' SVM-struct implementation for sequence structures.
 * 
 * @author eraldof
 * 
 */
public class NativeSVM implements StructuredAlgorithm {

	@Override
	public void train(ExampleInput[] inputs, ExampleOutput[] outputs,
			StringEncoding featureEncoding, StringEncoding stateEncoding) {
	}

	@Override
	public void train(ExampleInput[] inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInput[] inputsB,
			ExampleOutput[] outputsB, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {
	}

	native public double jniTest(double d, double[] dArray);

	static {
		// Library that implements the jniTest method.
		System.loadLibrary("svm_hmm_lib");
	}

}
