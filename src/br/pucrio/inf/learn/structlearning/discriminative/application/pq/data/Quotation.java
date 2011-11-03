package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

public class Quotation {
	private int[] quotationIndex;
	private int[][] coreferenceIndexes;
	
	public Quotation(int numberOfCoreferences) {
		quotationIndex = new int[2];
		coreferenceIndexes = new int[numberOfCoreferences][2];
	}
	
	public Quotation(Quotation quotation) {
		this(quotation.getNumberOfCoreferences());
		
		int[] auxQuotationIndex = quotation.getQuotationIndex();
		this.setQuotationIndex(auxQuotationIndex[0], auxQuotationIndex[1]);
		
		int numberOfCoreferences = quotation.getNumberOfCoreferences();
		for (int i = 0; i < numberOfCoreferences; ++i) {
			int[] auxCoreferenceIndex = quotation.getCoreferenceIndex(i);
			this.setCoreferenceIndex(i, auxCoreferenceIndex[0],
										auxCoreferenceIndex[1]);
		}
	}
	
	public void setQuotationIndex(int quotationStart, int quotationEnd) {
		quotationIndex[0] = quotationStart;
		quotationIndex[1] = quotationEnd;
	}
	
	public void setCoreferenceIndex(int coreferenceIndex, int coreferenceStart,
			int coreferenceEnd) {
		coreferenceIndexes[coreferenceIndex][0] = coreferenceStart;
		coreferenceIndexes[coreferenceIndex][1] = coreferenceEnd;
	}
	
	public int[] getQuotationIndex() {
		return quotationIndex;
	}
	
	public int[] getCoreferenceIndex(int coreferenceIndex) {
		return coreferenceIndexes[coreferenceIndex];
	}
	
	public int getNumberOfCoreferences() {
		return coreferenceIndexes.length;
	}
}
