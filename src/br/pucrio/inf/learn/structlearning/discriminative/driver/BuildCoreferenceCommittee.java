package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.json.JSONException;

import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

/**
 * Driver to build a committee of coreference models. The committee model is
 * just the mean of the given models.
 * 
 * @author eraldo
 * 
 */
public class BuildCoreferenceCommittee implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory
			.getLog(BuildCoreferenceCommittee.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder
				.withLongOpt("model")
				.withArgName("filename[,weight]")
				.hasArg()
				.isRequired()
				.withDescription(
						"File name with the model."
								+ " One can provide as many --model arguments as necessary."
								+ " Each model filename can include its corresponding weight.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("output")
				.withArgName("filename").hasArg().isRequired()
				.withDescription("Output model file name.").create());

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
		String[] modelFileNames = cmdLine.getOptionValues("model");
		String outputFileName = cmdLine.getOptionValue("output");

		CorefColumnDataset emptyDataset = new CorefColumnDataset();
		DPTemplateEvolutionModel finalModel = new DPTemplateEvolutionModel(0);
		try {

			for (String modelFileName : modelFileNames) {
				// Model filename and optional weight.
				String[] fileAndWeight = modelFileName.split(",");
				// Optional model weight.
				double weight = 1d;
				if (fileAndWeight.length > 1)
					weight = Double.parseDouble(fileAndWeight[1]);
				LOG.info(String.format("Loading model '%s' with weight %f...",
						fileAndWeight[0], weight));
				DPTemplateEvolutionModel memberModel = new DPTemplateEvolutionModel(
						fileAndWeight[0], emptyDataset, false);
				// Include model in the committee.
				finalModel.sumModel(memberModel, weight);
			}

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

		LOG.info(String.format("Committee model saved (%s)!", outputFileName));
	}

}
