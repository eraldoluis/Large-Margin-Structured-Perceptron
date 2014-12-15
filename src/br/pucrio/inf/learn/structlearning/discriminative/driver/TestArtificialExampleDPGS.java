package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.util.Map;

import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;

public class TestArtificialExampleDPGS {

	public static void main(String[] args) {

		// // Training options.
		// String trainPrefix =
		// "C:\\Users\\irving\\Desktop\\dpgs\\exemplos_artificial\\example";
		// String trainEdgeDatasetFileName = trainPrefix + ".edges";
		// String trainGPDatasetFileName = trainPrefix + ".grandparent";
		// String trainLSDatasetFileName = trainPrefix + ".siblings.left";
		// String trainRSDatasetFileName = trainPrefix + ".siblings.right";
		
		String templatesPrefix = "C:\\Users\\irving\\Desktop\\dpgs\\exemplos_artificial\\templates";
		String templatesEdgeFileName = templatesPrefix + ".edges";
		String templatesGPFileName = templatesPrefix + ".grandparent";
		String templatesLSFileName = templatesPrefix + ".siblings.left";
		String templatesRSFileName = templatesPrefix + ".siblings.right";

		final long testCacheSize = 4294967296l;

		final int numThreadToFillWeight = 2;

		// Test options.
		String testPrefix = "C:\\Users\\irving\\Desktop\\dpgs\\exemplos_artificial\\example";
		String testEdgeDatasetFilename = testPrefix + ".edges";
		String testGPDatasetFilename = testPrefix + ".grandparent";
		String testLSDatasetFilename = testPrefix + ".siblings.left";
		String testRSDatasetFilename = testPrefix + ".siblings.right";

		FeatureEncoding<String> featureEncoding = null;
		DPGSModel model;

		try {

			/*
			 * Create an empty and flexible feature encoding that will encode
			 * unambiguously all feature values. If the training dataset is big,
			 * this may not fit in memory and one should consider using a fixed
			 * encoding dictionary (based on test data or frequency on training
			 * data, for instance) or a hash-based encoding.
			 */
			featureEncoding = new StringMapEncoding();

			model = new DPGSModel(0);

			DPGSDataset testset = new DPGSDataset(new String[] {},
					new String[] {}, new String[] {}, "\\|", featureEncoding);
			
			String[] templatesFilename = new String[] {
					templatesEdgeFileName, templatesGPFileName,
					templatesLSFileName, templatesRSFileName };

			testset.loadExamplesAndGenerate(testEdgeDatasetFilename,
					testGPDatasetFilename, testLSDatasetFilename,
					testRSDatasetFilename, templatesFilename, model,
					10000000, "C:\\Users\\irving\\Desktop\\dpgs\\exemplos_artificial\\cache");
			
			testset.setModifierVariables();

			// Use dual inference algorithm for testing.
			// DPGSDualInference inferenceDual = new DPGSDualInference(
			// testset.getMaxNumberOfTokens());
			// inferenceDual
			// .setMaxNumberOfSubgradientSteps(maxSubgradientSteps);
			// inferenceDual.setBeta(beta);

			// // TODO test
			DPGSInference inferenceImpl = new DPGSInference(
					testset.getMaxNumberOfTokens(), numThreadToFillWeight);
			inferenceImpl.setCopyPredictionToParse(true);

			ExampleInputArray inputs = testset.getDPGSInputArray();

			int numberExamples = inputs.getNumberExamples();
			int[] inputToLoad = new int[numberExamples];

			for (int i = 0; i < inputToLoad.length; i++) {
				inputToLoad[i] = i;
			}
			DPGSOutput[] predicteds = new DPGSOutput[numberExamples];
			DPGSOutput[] outputs = testset.getOutputs();

			Map<Integer, AveragedParameter> p = model.getParameters();
			
			inputs.loadInOrder();

			for (int idx = 0; idx < numberExamples; ++idx)
				predicteds[idx] = (DPGSOutput) outputs[idx].createNewObject();

			for (int idx = 0; idx < numberExamples; ++idx) {
				// Predict (tag the output sequence).
				// LOG.info("Input: " + idx);
				DPGSInput input = (DPGSInput) inputs.get(idx);
				
				p.put(input.getGrandparentFeatures(2, 1, 0)[0], new AveragedParameter(100));
				p.put(input.getGrandparentFeatures(1, 2, 0)[0], new AveragedParameter(50));
				p.put(input.getGrandparentFeatures(1, 3, 0)[0], new AveragedParameter(20));
				p.put(input.getEdgeFeatures(0, 2)[0], new AveragedParameter(100));
				p.put(input.getEdgeFeatures(0, 1)[0], new AveragedParameter(20));
				p.put(input.getEdgeFeatures(0, 3)[0], new AveragedParameter(50));

				inferenceImpl.inference(model, input, predicteds[idx]);
			}

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
