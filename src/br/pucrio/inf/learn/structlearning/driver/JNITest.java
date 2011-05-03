package br.pucrio.inf.learn.structlearning.driver;

import br.pucrio.inf.learn.structlearning.algorithm.nsvm.NativeSVM;
import br.pucrio.inf.learn.structlearning.driver.Driver.Command;

public class JNITest implements Command {

	@Override
	public void run(String[] args) {
		NativeSVM svm = new NativeSVM();
		System.out.println("JNI test output: " + svm.jniTest(10d, null));
	}

}
