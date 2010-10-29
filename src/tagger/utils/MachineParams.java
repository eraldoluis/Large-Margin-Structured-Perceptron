package tagger.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;


public class MachineParams {
	Properties tempProp;
	MachineParams(String file) {
		
	try{	 
	 tempProp = new Properties();
	 tempProp.load(new FileInputStream(file));
	 } catch (IOException e) {
		 System.err.println("Could not load parameter file <"+file+">");
	 }
	}
	
	String get(String key)  throws Exception{
		String res = (String) tempProp.get(key);
		if(res==null) throw new Exception("Error getting config propertie for "+key);
		return res;
	}
}
