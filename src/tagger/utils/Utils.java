package tagger.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Vector;
import java.util.zip.GZIPInputStream;

import tagger.features.Fs;

public class Utils {

	public static final String GZIP_FILENAME_EXTENSION = ".gz";
	
	//get a BufferedReader from a fly and gunzip it on the fly if needed
	public static BufferedReader getBufferedReader(String filename, String encoding) throws FileNotFoundException, IOException {
	if(filename.endsWith(GZIP_FILENAME_EXTENSION))
		return new BufferedReader(new InputStreamReader((InputStream) new GZIPInputStream( new FileInputStream(filename)),encoding));
	else 
		return new BufferedReader(new InputStreamReader((InputStream) new FileInputStream(filename),encoding));
	}
	
	
	 public static void arraydump(double[][] delta, String s) {
		//dumping delta psi
		 for(int i=0;i<delta.length;++i)
		 	for(int j=0;j<delta[i].length;++j) 
		 		 System.err.println(s+"["+i+","+j+"]="+delta[i][j]);
	 }
	 public static void arraydump(int[][] delta, String s) {
			//dumping delta psi
			 for(int i=0;i<delta.length;++i)
			 	for(int j=0;j<delta[i].length;++j) 
			 		 System.err.println(s+"["+i+","+j+"]="+delta[i][j]);
		 }
	 
	 public static void arraydump(int[][] delta, String s,String out) {
	        try {
		    FileWriter fw = new FileWriter(out);	    
			//dumping delta psi
			 for(int i=0;i<delta.length;++i)
			 	for(int j=0;j<delta[i].length;++j) 
			 		 fw.write(s+"["+i+","+j+"]="+delta[i][j]+"\n");
			 fw.close();
	        }
	        catch(Exception e) {
	        	e.printStackTrace();
	        }
		 }
	 public static void arraydump(int[][][] delta, String s,String out) {
	        try {
		    FileWriter fw = new FileWriter(out);	    
			//dumping delta psi
			 for(int i=0;i<delta.length;++i)
			 	for(int j=0;j<delta[i].length;++j) 
			 		for(int k=0;k<delta[i][j].length;++k) 
			 		 fw.write(s+"["+i+","+j+","+k+"]="+delta[i][j][k]+"\n");
			 fw.close();
	        }
	        catch(Exception e) {
	        	e.printStackTrace();
	        }
		 }
	 
	 public static void arraydump(Vector<Vector< Vector<Integer> > > delta, String s,String out) {
	        try {
		    FileWriter fw = new FileWriter(out);	    
			//dumping delta psi
			 for(int i=0;i<delta.size();++i)
			 	for(int j=0;j<delta.get(i).size();++j) 
			 		for(int k=0;k<delta.get(i).get(j).size();++k) 
			 		 fw.write(s+"["+i+","+j+","+k+"]="+delta.get(i).get(j).get(k)+"\n");
			 fw.close();
	        }
	        catch(Exception e) {
	        	e.printStackTrace();
	        }
		 }
	 public static void arraydump(Fs[][] delta, String s) {
			//dumping delta psi
			 for(int i=0;i<delta.length;++i)
			 	for(int j=0;j<delta[i].length;++j) 
			 		 System.err.println(s+"["+i+","+j+"]="+delta[i][j]);
		 }
}
