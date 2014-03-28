package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.json.JSONException;

import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CoreferenceMaxBranchInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CoreferenceMaxBranchInference.InferenceStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.FeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.SimpleFeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

/**
 * Driver to apply a coreference model to a given corpus and, optionally,
 * evaluating the result.
 * 
 * @author eraldo
 * 
 */
public class PrintCorefNonSingularClusters implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory
			.getLog(PrintCorefNonSingularClusters.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("model")
				.withArgName("filename").hasArg().isRequired()
				.withDescription("File name with the model.").create());
		options.addOption(OptionBuilder.withLongOpt("test").isRequired()
				.hasArg().withArgName("filename")
				.withDescription("Test dataset file name.").create());
		options.addOption(OptionBuilder.withLongOpt("onlycorrect")
				.withDescription("Print only correctly predicted documents.")
				.create());

		// Parse the command-line arguments.
		CommandLine cmdLine = null;
		PosixParser parser = new PosixParser();
		try {
			cmdLine = parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println(e.getMessage());
			CommandLineOptionsUtil.usage(getClass().getSimpleName(), options);
		}

		// Print the list of options along the values provided by the user.
		CommandLineOptionsUtil.printOptionValues(cmdLine, options);

		/*
		 * Get the options given in the command-line or the corresponding
		 * default values.
		 */
		String modelFileName = cmdLine.getOptionValue("model");
		String testDatasetFileName = cmdLine.getOptionValue("test");
		boolean onlyCorrect = cmdLine.hasOption("onlycorrect");

		CorefColumnDataset testDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		MapEncoding<Feature> tplFeatureEnconding = null;
		try {
			/*
			 * Create an empty and flexible feature encoding that will encode
			 * unambiguously all feature values. If the training dataset is big,
			 * this may not fit in memory.
			 */
			featureEncoding = new StringMapEncoding();

			LOG.info("Loading dataset...");
			testDataset = new CorefColumnDataset(featureEncoding,
					(Collection<String>) null);
			testDataset.setCheckMultipleTrueEdges(false);
			testDataset.load(testDatasetFileName);

			tplFeatureEnconding = testDataset.getExplicitEncoding();

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		LOG.info("Loading model and templates...");
		DPTemplateEvolutionModel model = null;
		try {
			model = new DPTemplateEvolutionModel(modelFileName, testDataset,
					false);
		} catch (JSONException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		} catch (IOException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		} catch (DatasetException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		}

		int max = 30;
		LOG.info("Finding largest parameters...");
		PriorityQueue<ParameterWeight> weights = new PriorityQueue<ParameterWeight>();
		for (Entry<Integer, AveragedParameter> entry : model.getParameters()
				.entrySet()) {
			weights.add(new ParameterWeight(entry.getKey(), entry.getValue()
					.get()));
			if (weights.size() > max)
				weights.poll();
		}

		FeatureTemplate[] tpls = testDataset.getTemplates()[0];
		for (ParameterWeight pw : weights) {
			Feature ftr = tplFeatureEnconding.getValueByCode(pw.parameter);
			SimpleFeatureTemplate tpl = (SimpleFeatureTemplate) tpls[ftr
					.getTemplateIndex()];
			String ftrStr = "";
			int[] vals = ftr.getValues();
			for (int idxVal = 0; idxVal < vals.length; ++idxVal) {
				ftrStr += testDataset
						.getFeatureLabel(tpl.getFeatures()[idxVal]);
				ftrStr += "=" + featureEncoding.getValueByCode(vals[idxVal]);
				ftrStr += " ";
			}
			System.out.println("Parameter " + ftrStr + " with weight "
					+ pw.weight);
		}

		LOG.info("Generating features from templates...");
		testDataset.generateFeatures();

		// Inference algorithm.
		CoreferenceMaxBranchInference inference = new CoreferenceMaxBranchInference(
				testDataset.getMaxNumberOfTokens(), 0,
				InferenceStrategy.LBRANCH);

		/*
		 * Model application.
		 */
		DPInput[] inputs = testDataset.getInputs();
		DPOutput[] outputs = testDataset.getOutputs();

		// Allocate predicted output structures.
		DPOutput[] predicteds = new DPOutput[inputs.length];
		for (int idx = 0; idx < inputs.length; ++idx)
			predicteds[idx] = inputs[idx].createOutput();

		/*
		 * Fill the list of predicted outputs and the latent trees for the
		 * correct outputs.
		 */
		int numExsCorrects = 0;
		int numExsWithNonSingularClusters = 0;
		for (int idx = 0; idx < inputs.length; ++idx) {
			// Predict (tag the output sequence).
			inference.inference(model, inputs[idx], predicteds[idx]);
			// Constrained prediction: latent structures.
			inference.partialInference(model, inputs[idx], outputs[idx],
					outputs[idx]);

			if (onlyCorrect
					&& !Arrays.equals(
							predicteds[idx].getInvertedBranchingArray(),
							outputs[idx].getInvertedBranchingArray()))
				// Skip incorrect predictions when required.
				continue;

			++numExsCorrects;

			// Check how many non-singular clusters there are in this output.
			Map<Integer, ? extends Set<Integer>> explicitClusters = CoreferenceMaxBranchInference
					.createExplicitClustering((CorefOutput) predicteds[idx]);

			CorefOutput predicted = (CorefOutput) predicteds[idx];
			Set<Integer> nonSingularClusters = new TreeSet<Integer>();
			int numMentions = predicted.size();
			int[] numberOfChildren = new int[numMentions];
			for (int m = 1; m < numMentions; ++m) {
				int parent = predicted.getInvertedBranchingArray()[m];
				if (parent != 0) {
					++numberOfChildren[parent];
					if (numberOfChildren[parent] > 1)
						nonSingularClusters.add(predicted.getClusterId(parent));
				}
			}

			// Print non-singular clusters.
			if (nonSingularClusters.size() > 0) {
				++numExsWithNonSingularClusters;
				System.out.println("*** Non-singular clusters of example "
						+ idx + " ***");
				for (int id : nonSingularClusters) {
					System.out.print("Non-singular cluster tree " + id + ": ");
					Set<Integer> cluster = explicitClusters.get(id);
					for (int m : cluster) {
						int parent = predicted.getInvertedBranchingArray()[m];
						System.out.print("(" + parent + "," + m + "), ");
					}
					System.out.println();
				}
				System.out.println();
			}
		}

		System.out.println("# completely correct examples: " + numExsCorrects
				+ "\n");
		System.out
				.println("# completely correct examples with non-singular clusters: "
						+ numExsWithNonSingularClusters + "\n");

		LOG.info("Model application done!");
	}

	private static class ParameterWeight implements Comparable<ParameterWeight> {
		public int parameter;
		public double weight;

		public ParameterWeight(int parameter, double weight) {
			this.parameter = parameter;
			this.weight = weight;
		}

		@Override
		public int compareTo(ParameterWeight o) {
			double w = Math.abs(weight);
			double wo = Math.abs(o.weight);
			if (w < wo)
				return -1;
			if (w > wo)
				return 1;
			return 0;
		}

	}
}
