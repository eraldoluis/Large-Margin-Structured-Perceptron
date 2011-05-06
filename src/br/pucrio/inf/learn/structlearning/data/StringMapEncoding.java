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
public class StringMapEncoding extends MapEncoding<String> {

	public StringMapEncoding() {
		super();
	}

	public StringMapEncoding(BufferedReader reader) throws IOException {
		super(reader);
	}

	public StringMapEncoding(InputStream is) throws IOException {
		super(is);
	}

	public StringMapEncoding(String fileName) throws IOException {
		super(fileName);
	}

	public StringMapEncoding(String[] values) {
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
