package tagger.learning;

import java.io.PrintStream;
import java.util.Vector;

public class Inclass_statistics {
	 public
		 double [][] M_nobj, M_nans, M_nfull, M_R, M_P, M_F; 
	 
	 public void dump(PrintStream cout) {
		 LearningStatistics.dumpArraydouble(cout,"M_nobj",M_nobj);
		 LearningStatistics.dumpArraydouble(cout,"M_nans",M_nans);
		 LearningStatistics.dumpArraydouble(cout,"M_nfull",M_nfull);
		 LearningStatistics.dumpArraydouble(cout,"M_R",M_R);
		 LearningStatistics.dumpArraydouble(cout,"M_P",M_P);
		 LearningStatistics.dumpArraydouble(cout,"M_F",M_F);
	 }
	 
	 public Inclass_statistics(int CV, int T) {
		  M_nobj=new double[CV][T];
		  M_nans=new double[CV][T];
		  M_nfull=new double[CV][T];
		  M_R=new double[CV][T];
		  M_P=new double[CV][T];
		  M_F=new double[CV][T];
	 }
}
