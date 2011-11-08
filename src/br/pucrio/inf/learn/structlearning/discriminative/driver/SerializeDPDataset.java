package br.pucrio.inf.learn.structlearning.discriminative.driver;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPDataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.HybridStringEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.JavaHashCodeEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Lookup3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur2Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

/**
 * Serialize a dataset using a given encoding to speedup loading before
 * training.
 * 
 * @author eraldo
 * 
 */
public class SerializeDPDataset implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(SerializeDPDataset.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("in").isRequired()
				.withArgName("filename").hasArg()
				.withDescription("Input filename.").create());
		options.addOption(OptionBuilder.withLongOpt("out")
				.withArgName("filename").hasArg()
				.withDescription("Output filename.").create());
		options.addOption(OptionBuilder
				.withLongOpt("encoding")
				.withArgName("filename")
				.hasArg()
				.withDescription(
						"Filename that contains a list of considered feature"
								+ " values. Any feature value not present in"
								+ " this file is ignored.").create());
		options.addOption(OptionBuilder
				.withLongOpt("minfreq")
				.withArgName("integer")
				.hasArg()
				.withDescription(
						"Minimum frequency of feature values in the encoding "
								+ "file used to cutoff low frequent values.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("murmur")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Murmur3 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder.withLongOpt("hashseed")
				.withArgName("seed").hasArg()
				.withDescription("Seed for the hash-based encodings.").create());
		options.addOption(OptionBuilder
				.withLongOpt("murmur3")
				.withArgName("size")
				.hasArg()
				.withDescription(
						"Use a Murmur3 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("murmur2")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Murmur2 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lookup3")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Lookup3 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("javahash")
				.withArgName("hash table size")
				.hasArg()
				.withDescription(
						"Use the default Java hashing function (hashCode method) "
								+ "to encode feature values.").create());

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
		String inputFilename = cmdLine.getOptionValue("in");
		String outputFilename = cmdLine.getOptionValue("out");
		String encodingFile = cmdLine.getOptionValue("encoding");
		String minFreqStr = cmdLine.getOptionValue("minfreq");
		String hashSeed = cmdLine.getOptionValue("hashseed");
		String murmur = cmdLine.getOptionValue("murmur");
		String murmur3 = cmdLine.getOptionValue("murmur3");
		String murmur2 = cmdLine.getOptionValue("murmur2");
		String lookup3 = cmdLine.getOptionValue("lookup3");
		String javaHashSizeStr = cmdLine.getOptionValue("javahash");

		DPDataset inDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		FeatureEncoding<String> additionalFeatureEncoding = null;
		try {

			// Create (or load) the feature value encoding.
			if (encodingFile != null) {

				if (minFreqStr == null) {
					/*
					 * Load a map-based encoding from the given file. Thus, the
					 * feature values present in this file will be encoded
					 * unambiguously but any unknown value will be ignored.
					 */
					LOG.info("Loading encoding file...");
					featureEncoding = new StringMapEncoding(encodingFile);
				} else {
					/*
					 * Load map-based encoding from the given file and filter
					 * out low frequent feature values according to feature
					 * frequencies given in the file.
					 */
					LOG.info("Loading encoding file...");
					int minFreq = Integer.parseInt(minFreqStr);
					featureEncoding = new StringMapEncoding(encodingFile,
							minFreq);
				}

			} else if (minFreqStr != null) {
				LOG.error("minfreq=? only works together with option encoding");
				System.exit(1);
			}

			/*
			 * Additional feature encoding (or the only one, if a fixed encoding
			 * file is not given).
			 */
			if (murmur != null) {

				// Create a feature encoding based on the Murmur3 hash function.
				int size = TrainDP.parseValueDirectOrBits(murmur);
				if (hashSeed == null)
					additionalFeatureEncoding = new Murmur3Encoding(size);
				else
					additionalFeatureEncoding = new Murmur3Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (murmur3 != null) {

				// Create a feature encoding based on the Murmur3 hash function.
				int size = TrainDP.parseValueDirectOrBits(murmur3);
				if (hashSeed == null)
					additionalFeatureEncoding = new Murmur3Encoding(size);
				else
					additionalFeatureEncoding = new Murmur3Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (murmur2 != null) {

				// Create a feature encoding based on the Murmur2 hash function.
				int size = TrainDP.parseValueDirectOrBits(murmur2);
				if (hashSeed == null)
					additionalFeatureEncoding = new Murmur2Encoding(size);
				else
					additionalFeatureEncoding = new Murmur2Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (lookup3 != null) {

				// Create a feature encoding based on the Lookup3 hash function.
				int size = TrainDP.parseValueDirectOrBits(lookup3);
				if (hashSeed == null)
					additionalFeatureEncoding = new Lookup3Encoding(size);
				else
					additionalFeatureEncoding = new Lookup3Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (javaHashSizeStr != null) {

				// Create a feature encoding based on the Java hash function.
				additionalFeatureEncoding = new JavaHashCodeEncoding(
						TrainDP.parseValueDirectOrBits(javaHashSizeStr));

			}

			if (featureEncoding == null) {

				if (additionalFeatureEncoding == null)
					/*
					 * No encoding given by the user. Create an empty and
					 * flexible feature encoding that will encode unambiguously
					 * all feature values. If the training dataset is big, this
					 * may not fit in memory.
					 */
					featureEncoding = new StringMapEncoding();
				else
					// Only one feature encoding given.
					featureEncoding = additionalFeatureEncoding;

			} else if (additionalFeatureEncoding != null)
				/*
				 * The user specified two encodings. Combine them in one hybrid
				 * encoding.
				 */
				featureEncoding = new HybridStringEncoding(featureEncoding,
						additionalFeatureEncoding);

			LOG.info("Feature encoding: "
					+ featureEncoding.getClass().getSimpleName());

			LOG.info("Loading input dataset...");
			inDataset = new DPDataset(featureEncoding);
			inDataset.load(inputFilename);

			LOG.info("Feature encoding size: " + featureEncoding.size());

			LOG.info("Serializing dataset...");
			inDataset.serialize(outputFilename);

			LOG.info("Serialization successfully done!");

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}
	}
}
