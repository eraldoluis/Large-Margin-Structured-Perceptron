package br.pucrio.inf.learn.util;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

/**
 * Utilitary procedures to work with command-line options.
 * 
 * @author eraldof
 * 
 */
public class CommandLineOptionsUtil {

	/**
	 * Print a usage message and abort the execution.
	 * 
	 * @param syntaxLine
	 * @param ops
	 */
	public static void usage(String syntaxLine, Options ops) {
		HelpFormatter hf = new HelpFormatter();
		hf.setWidth(Integer.MAX_VALUE);
		hf.setOptionComparator(null);
		hf.printHelp(syntaxLine, ops, true);
		System.exit(1);
	}

	/**
	 * Print the list of options and their values.
	 * 
	 * @param cmdLine
	 * @param options
	 */
	public static void printOptionValues(CommandLine cmdLine, Options options) {
		for (Object obj : options.getOptions()) {
			Option op = (Option) obj;
			String name;
			String value = ":";
			String[] values;
			name = op.hasLongOpt() ? op.getLongOpt() : op.getOpt();
			if (op.hasArg()) {
				values = cmdLine.getOptionValues(name);
				if (values != null)
					for (String val : values)
						value += " " + val;
			} else
				value += " " + Boolean.toString(cmdLine.hasOption(name));
			System.out.println("\t" + name + value);
		}
	}

}
