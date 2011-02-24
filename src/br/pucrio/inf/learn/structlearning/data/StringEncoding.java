package br.pucrio.inf.learn.structlearning.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

/**
 * Encoding for string values.
 * 
 * @author eraldo
 *
 */
public class StringEncoding extends Encoding<String> {

	@Override
	public void load(BufferedReader reader) throws IOException {
		String val;
		while ((val = reader.readLine()) != null)
			putValue(val);
	}

	@Override
	public void save(PrintStream ps) {
		for (String val : getOrderedValues())
			ps.println(val);
	}

}
