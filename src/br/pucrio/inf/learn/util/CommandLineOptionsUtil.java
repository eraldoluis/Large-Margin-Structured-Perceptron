package br.pucrio.inf.learn.util;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class CommandLineOptionsUtil {

	public static void usage(String syntaxLine, Options ops) {
		HelpFormatter hf = new HelpFormatter();
		hf.setOptionComparator(null);
		hf.printHelp(syntaxLine, ops, true);
		System.exit(1);
	}

	public static void printOptionValues(CommandLine cmdLine, Options options) {
		for (Object obj : options.getOptions()) {
			Option op = (Option) obj;
			String name;
			String value = ":";
			String[] values;
			if (op.hasLongOpt()) {
				name = op.getLongOpt();
				values = cmdLine.getOptionValues(op.getLongOpt());
			} else {
				name = op.getOpt();
				values = cmdLine.getOptionValues(op.getOpt());
			}
			if (values != null)
				for (String val : values)
					value += " " + val;
			System.out.println("\t" + name + value);
		}
	}

}
