package br.pucrio.inf.learn.structlearning.hadoop.driver;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import br.pucrio.inf.learn.mr.util.CommandDriverTool;

/**
 * Driver for Hadoop commands.
 * 
 * @author eraldof
 * 
 */
public class HadoopDriver extends Configured implements Tool {

	public int run(String[] args) throws Exception {

		// The general driver.
		CommandDriverTool driver = new CommandDriverTool(getConf());

		driver.addCommand("TrainHmm", new TrainHmmOnHadoopMain(),
				"Train an HMM model using an online algorithm.");

		driver.run(args);

		return 0;

	}

	public static void main(String[] args) throws Exception {
		// Let ToolRunner handle generic command-line options
		int res = ToolRunner.run(new Configuration(), new HadoopDriver(), args);
		System.exit(res);
	}

}
