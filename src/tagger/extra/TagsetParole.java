package tagger.extra;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.HashMap;

/**
 * A class to map Catalan parole tags to attributes (for CONLL format)
 * @author jordi
 *
 */
public class TagsetParole {
	HashMap<features,String> shorts;
	
	HashMap<pos, features[]> mapfeatures;
	final static int posLength=2;
	
	public TagsetParole() {
		 mapfeatures= new HashMap<pos, features[]>();
		 shorts = new HashMap<features,String>();
		 shorts.put(features.gender,"gen");
		 shorts.put(features.person,"per");
		 shorts.put(features.mod,"mod");
		 shorts.put(features.number,"num");
		 shorts.put(features.tense,"ten");
		 shorts.put(features.spfor,"for");
		 shorts.put(features.pos,"pos");
		 shorts.put(features.cas,"cas");
		 shorts.put(features.pol,"pol");
		 shorts.put(features.function,"fun");
}
	
	enum tense  { none, 
		conditional, //c, //conditional, 
		future, // f, //future 
		imperfect, //imperfect, i, //imperfect, 
		past, //s, //past
		present //p //present 
	}
	
	enum mod { none,
		gerund, //" => "mod=g",
	    imperative, // " => "mod=m",
	    indicative , // , // => "mod=i", 
	    infinitive , // => "mod=n",
	    pastparticiple, //  => "mod=p",
	    subjunctive// => "mod=s",
	}
	
	enum cas {
		nominative,//" => "cas=n",
		oblique,//" => "cas=o",
		accusative,//" =>  "cas=a",
		dative,//" => "cas=d",
	}
	 enum features { none, number, pol, cas, person,  pos, gender,   mod,tense,   spfor, function}
	 enum number { none, singular, plural, common}
	 enum gender { none, masculine, femenine, neutral, common}
	 enum person { none, first, second, third}
	 
	 
	 enum pos {
		 ao,
		 aq,
		 cc,
		 cs,
		 da,
		 dd,
		 de,
		 di,
		 dn,
		 dp,
		 dr,
		 dt,
		 fa,
		 fc,
		 fd,
		 fe,
		 fg,
		 fh,
		 fi,
		 fp,
		 fs,
		 fl,
		 fx,
		 ft,
		 fz,
		 nc,
		 np,
		 p0,
		 pd,
		 pi,
		 pn,
		 pp,
		 pr,
		 pt,
		 px,
		 rg,
		 rn,
		 sp,
		 va,
		 vm,
		 vs,
		 zm,
		 zp,
		 zu,
		 i, //?
		 w, //?
		 z, //?
	 }
	 
	 public static String mainPos(String tag) {
		 return tag.length()>posLength ? tag.substring(0,posLength): tag;
	 }
	 
	 public String tag2Features(String utag) {
		 StringBuffer res = new StringBuffer();
		 String tag = utag.toLowerCase();
		 try {
		 pos fpos = pos.valueOf( mainPos(tag));
		 features[] posfeat= mapfeatures.get(pos.valueOf( mainPos(tag)));
		 
		 
		          
		if(posfeat!=null) {
		 char[] mf= new char[features.values().length];
		 for(int i=0;i<posfeat.length;++i) {
			 features f = posfeat[i];
			 if(f!=features.none && tag.charAt(posLength+i)!='0') {
				mf[f.ordinal()] = tag.charAt(posLength+i);
			 }
		 }
		 
		 for(features fa: features.values()) {
			 char c = mf[fa.ordinal()];
			 if(c!='\u0000') {
			  if(res.length()>0) res.append("|");
			   res.append(shorts.get(fa)+"="+c);
			 }
		 }
		}
		} catch(Exception e) {
			e.printStackTrace();
		}
		 
		 if(res.length()==0) {return "_";}
		 return res.toString();
	 }
	 
	 /**
	  * loads a map form a file
	  * @param dname
	  */
	 void loadMap(String dname) {
		 try {String buff;
			BufferedReader fin = new BufferedReader(new FileReader(dname));
			try {
				while((buff=fin.readLine())!=null) {	
					String[] fields= buff.split("\t");
					
					int fl =  fields.length-posLength;
					if(fl<0) fl=0;
					
					features featpos[] = new features[fl];
					for(int i=posLength;i<fields.length;++i)  {
						featpos[i-posLength]=features.valueOf(fields[i]);
					}
					
					mapfeatures.put(pos.valueOf(fields[0]), featpos);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	 }
	 
	 /**
	  * 
	  * reads a multitag file and converts it to CONLL format
	  * uses the map in file  parole.txt
	  * 
	  * @param args 0: input file (stdin if ommited)
	  * @param args 1: output file (stdout if ommited)
	  * @throws IOException
	  */
	 public static void main(String [] args) throws IOException {
		 TagsetParole  tagset = new TagsetParole();
		 tagset.loadMap("./parole.txt");
		 String buff;
		 InputStreamReader infile;
		 PrintStream outfile;
		 
		 if(args.length >0) infile =new InputStreamReader( new FileInputStream(args[0]));
		 else infile=new InputStreamReader(System.in);
		 
		 if(args.length >1) outfile =new PrintStream(args[1]);
		 else outfile=new PrintStream(System.out);
		 
		 int ntoken=1;
		 BufferedReader fin = new BufferedReader(infile);
		 while((buff=fin.readLine())!=null) {	
			 if(buff.startsWith("%%#")) { 
				 if(buff.charAt(3)=='S') {outfile.println(); ntoken=1;}
				 outfile.println(buff);
			 }
			 else {
			 //split with tabular second  field is POS
			 String fields[] = buff.split("[ \t]");
			 String pos = fields[2].toLowerCase();
			 // CONLL extrange upper lower case conversion (fix)
			 if(pos.charAt(0)=='f' || pos.length()==1) {pos = fields[2].charAt(0)+pos.substring(1,pos.length());}
			 
			 outfile.print(ntoken+"\t"+fields[0]+"\t"+fields[1]+"\t"+pos.charAt(0)+"\t"+mainPos(pos)+"\t"+tagset.tag2Features(pos));
			 outfile.println();
			 ntoken++;
			 }
		 }
		 outfile.close();
	 }
}
