package br.pucrio.inf.learn.structlearning.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;

/**
 * Encoding for string values.
 * 
 * @author eraldo
 * 
 */
public class StringEncoding extends Encoding<String> {

	public StringEncoding() {
		super();
	}

	public StringEncoding(BufferedReader reader) throws IOException {
		super(reader);
	}

	public StringEncoding(InputStream is) throws IOException {
		super(is);
	}

	public StringEncoding(String fileName) throws IOException {
		super(fileName);
	}

	public StringEncoding(String[] values) {
		super(values);
	}

	@Override
	public void load(BufferedReader reader) throws IOException {
		String val;
		while ((val = reader.readLine()) != null)
			put(val);
	}

	@Override
	public void save(PrintStream ps) {
		for (String val : getValues())
			ps.println(val);
	}

}
