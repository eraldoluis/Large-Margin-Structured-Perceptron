package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.evaluate;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;

public class Accuracy implements Metric {

	private String script;
	private String conllGolden;
	private String conllPredicted;
	private boolean quiet;
	private DPGSDataset testset;
	private final static Log LOG = LogFactory.getLog(Accuracy.class);

	public Accuracy(String script, String conllGolden,
			String conllPredicted, boolean quiet, DPGSDataset testset) {
		super();
		this.script = script;
		this.conllGolden = conllGolden;
		this.conllPredicted = conllPredicted;
		this.quiet = quiet;
		this.testset = testset;
	}

	@Override
	public void evaluate(int epoch, DPGSOutput[] corrects,
			DPGSOutput[] predicteds) {
		try {
			// Delete previous epoch output file if it exists.
			File o = new File(conllPredicted);
			if (o.exists())
				o.delete();

			LOG.info(String
					.format("Saving input CoNLL file (%s) to output file (%s) with predicted columns",
							conllGolden, conllPredicted));

			testset.save(conllGolden, conllPredicted, predicteds);

			try {
				LOG.info("Evaluation after epoch " + epoch + ":");
				// Execute CoNLL evaluation scripts.
				evaluateWithConllScripts(script, conllGolden,
						conllPredicted, quiet);
			} catch (Exception e) {
				LOG.error("Running evaluation scripts", e);
			}

		} catch (IOException e) {
			LOG.error("Saving test file with predicted column", e);
		} catch (DatasetException e) {
			LOG.error("Saving test file with predicted column", e);
		}
	}
	
	public static void evaluateWithConllScripts(String script,
			String conllGolden, String conllPredicted, boolean quiet)
			throws IOException, CommandException, InterruptedException {
		// Command to evaluate the predicted information.
		String cmd = String.format("perl %s -g %s -s %s%s", script,
				conllGolden, conllPredicted, (quiet ? " -q" : ""));
		execCommandAndRedirectOutputAndError(cmd, null);
	}

	/**
	 * Execute the given system command and redirects its standard and error
	 * outputs to the standard and error outputs of the JVM process.
	 * 
	 * @param command
	 * @param path
	 * @throws IOException
	 * @throws CommandException
	 * @throws InterruptedException
	 */
	private static void execCommandAndRedirectOutputAndError(String command,
			File path) throws IOException, CommandException,
			InterruptedException {
		String line;

		// Execute command.
		LOG.info("Running command: " + command);
		Process p = Runtime.getRuntime().exec(command, null, path);

		// Redirect standard output of process.
		BufferedReader out = new BufferedReader(new InputStreamReader(
				p.getInputStream()));
		while ((line = out.readLine()) != null)
			System.out.println(line);
		out.close();

		// Redirect error output of process.
		BufferedReader error = new BufferedReader(new InputStreamReader(
				p.getErrorStream()));
		while ((line = error.readLine()) != null)
			System.err.println(line);
		error.close();

		if (p.waitFor() != 0)
			throw new CommandException("Command exit with non-zero status");
	}
	
	private static class CommandException extends Exception {
		/**
		 * Auto-generated serial version ID.
		 */
		private static final long serialVersionUID = 6582860853130630178L;

		public CommandException(String message) {
			super(message);
		}
	}

}