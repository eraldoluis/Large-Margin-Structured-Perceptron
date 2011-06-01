package br.pucrio.inf.learn.structlearning.discriminative.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Combine a closed encoding with an additional encoding.
 * 
 * An encoding is closed when it is not able to represent any string value,
 * i.e., it eventually returns a negative code (invalid) for some values. This
 * hybrid implementation uses an additional encoding to deal with this missing
 * values. The additional encoding can be closed or open.
 * 
 * @author eraldof
 * 
 */
public class HybridStringEncoding implements FeatureEncoding<String> {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory
			.getLog(HybridStringEncoding.class);

	/**
	 * Implementation that is not able to represent any string value, i.e., for
	 * some values, it will return a negative value.
	 */
	private FeatureEncoding<String> encodingClosed;

	/**
	 * Additional implementations that handles string values not represented by
	 * the closed encoding. This encoding can be closed or open.
	 */
	private FeatureEncoding<String> encodingAdditional;

	/**
	 * The closed encoding must have a fixed size.
	 */
	private int sizeClosed;

	/**
	 * Create a hybrid encoding using the two given encodings.
	 * 
	 * @param closed
	 * @param additional
	 */
	public HybridStringEncoding(FeatureEncoding<String> closed,
			FeatureEncoding<String> additional) {
		this.encodingClosed = closed;
		this.encodingAdditional = additional;
		this.sizeClosed = closed.size();
	}

	@Override
	public int size() {
		return sizeClosed + encodingAdditional.size();
	}

	@Override
	public int put(String value) {
		int code = encodingClosed.put(value);
		if (code < 0) {
			code = encodingAdditional.put(value);
			if (code < 0)
				return code;
			code += sizeClosed;
		}
		return code;
	}

	@Override
	public int getCodeByValue(String value) {
		int code = encodingClosed.getCodeByValue(value);
		if (code < 0) {
			code = encodingAdditional.getCodeByValue(value);
			if (code < 0)
				return code;
			code += sizeClosed;
		}
		return code;
	}

	@Override
	public String getValueByCode(int code) {
		if (code < sizeClosed)
			return encodingClosed.getValueByCode(code);
		return encodingAdditional.getValueByCode(code - sizeClosed);
	}

	@Override
	public Collection<Integer> getCodes() {
		return null;
	}

	@Override
	public Collection<String> getValues() {
		return null;
	}

	@Override
	public void setReadOnly(boolean value) {
		encodingClosed.setReadOnly(value);
		encodingAdditional.setReadOnly(value);
	}

	@Override
	public void load(InputStream is) throws IOException {
		throw new IOException("Operation not supported");
	}

	@Override
	public void load(String fileName) throws IOException {
		throw new IOException("Operation not supported");
	}

	@Override
	public void load(BufferedReader reader) throws IOException {
		throw new IOException("Operation not supported");
	}

	@Override
	public void save(OutputStream os) {
		LOG.error("Operation not supported");
	}

	@Override
	public void save(String fileName) throws FileNotFoundException {
		LOG.error("Operation not supported");
	}

	@Override
	public void save(PrintStream ps) {
		LOG.error("Operation not supported");
	}

}
