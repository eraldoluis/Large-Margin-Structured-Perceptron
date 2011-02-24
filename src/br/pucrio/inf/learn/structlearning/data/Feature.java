package br.pucrio.inf.learn.structlearning.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

public interface Feature extends Comparable<Feature> {

	public void load(BufferedReader reader) throws IOException;

	public void save(PrintStream ps);

}
