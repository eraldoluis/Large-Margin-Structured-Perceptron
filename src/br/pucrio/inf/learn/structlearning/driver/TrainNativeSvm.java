package br.pucrio.inf.learn.structlearning.driver;

import br.pucrio.inf.learn.structlearning.algorithm.nsvm.NativeSvm;
import br.pucrio.inf.learn.structlearning.driver.Driver.Command;

public class TrainNativeSvm implements Command {

	@Override
	public void run(String[] args) {
		NativeSvm svm = new NativeSvm();
	}

}
