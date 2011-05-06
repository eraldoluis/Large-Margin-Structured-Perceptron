package br.pucrio.inf.learn.structlearning.algorithm.nsvm;

import br.pucrio.inf.learn.structlearning.algorithm.StructuredAlgorithm;
import br.pucrio.inf.learn.structlearning.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.task.Model;

/**
 * Stub for Joachims' SVM-struct implementation for sequence structures.
 * 
 * @author eraldof
 * 
 */
public class NativeSvm implements StructuredAlgorithm {

	private boolean partiallyAnnotatedExamples;

	private TrainingListener listener;

	private Model model;

	/**
	 * Used by the native code to store the pointer to loaded examples.
	 */
	private long nativePointer;

	/**
	 * Load the native library that includes the SVM implementation.
	 */
	static {
		// Library that implements the jniTest method.
		System.loadLibrary("svm_hmm_lib");
	}

	@Override
	public void train(ExampleInput[] inputs, ExampleOutput[] outputs,
			StringMapEncoding featureEncoding, StringMapEncoding stateEncoding) {
	}

	@Override
	public void train(ExampleInput[] inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInput[] inputsB,
			ExampleOutput[] outputsB, StringMapEncoding featureEncoding,
			StringMapEncoding stateEncoding) {
	}

	/**
	 * Init the structures used by the native solver in the native memory and
	 * load the training examples. Store a pointer to theses structures in the
	 * <code>nativePointer</code> attribute.
	 * 
	 * @param inputs
	 * @return
	 */
	native protected int initNativeStructures(SequenceInput[] inputs);

	/**
	 * Update the output structures of the loaded examples.
	 * 
	 * @param outputs
	 * @return
	 */
	native protected int updateOutputStructures(SequenceOutput[] outputs);

	/**
	 * Free the previously allocated native structures.
	 * 
	 * @return
	 */
	native protected int freeNativeStructures();

	/**
	 * Train a model using the current examples and the given parameters (c and
	 * eps).
	 * 
	 * @param c
	 * @param eps
	 * @return
	 */
	native protected int train(double c, double eps);

	@Override
	public Model getModel() {
		return model;
	}

	@Override
	public void setPartiallyAnnotatedExamples(boolean value) {
		partiallyAnnotatedExamples = value;
	}

	@Override
	public void setListener(TrainingListener listener) {
		this.listener = listener;
	}

}
