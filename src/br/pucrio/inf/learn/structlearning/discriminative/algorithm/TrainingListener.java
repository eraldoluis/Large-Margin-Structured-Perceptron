package br.pucrio.inf.learn.structlearning.discriminative.algorithm;

import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Interface for listeners that observe a training algorithm.
 * 
 * @author eraldof
 * 
 */
public interface TrainingListener {

	/**
	 * Called before starting the training procedure.
	 * 
	 * @param impl
	 *            task-specific inference algorithms.
	 * @param curModel
	 *            the current model (no averaging).
	 * 
	 * @return <code>false</code> to not start the training procedure.
	 */
	boolean beforeTraining(Inference impl, Model curModel);

	/**
	 * Called after the training procedure ends.
	 * 
	 * @param impl
	 *            task-specific inference algorithms.
	 * @param curModel
	 *            the current model (no averaging).
	 */
	void afterTraining(Inference impl, Model curModel);

	/**
	 * Called before starting an epoch (processing the whole training set).
	 * 
	 * @param impl
	 *            task-specific inference algorithms.
	 * @param curModel
	 *            the current model (no averaging).
	 * @param epoch
	 *            the current epoch (starts in zero).
	 * @param iteration
	 *            current iteration (number of inference/update steps).
	 * 
	 * @return <code>false</code> to stop the training procedure.
	 */
	boolean beforeEpoch(Inference impl, Model curModel, int epoch, int iteration);

	/**
	 * Called after an epoch (processing the whole training set).
	 * 
	 * @param impl
	 *            task-specific inference algorithms.
	 * @param curModel
	 *            the current model (no averaging).
	 * @param epoch
	 *            the current epoch (starts in zero).
	 * @param loss
	 *            the training set loss during accumulated during the last
	 *            epoch.
	 * @param iteration
	 *            current iteration (number of inference/update steps).
	 * 
	 * @return
	 */
	boolean afterEpoch(Inference impl, Model curModel, int epoch, double loss,
			int iteration);

	/**
	 * Called on each progress report.
	 * 
	 * @param impl
	 * @param curModel
	 * @param epoch
	 * @param loss
	 * @param iteration
	 */
	void progressReport(Inference impl, Model curModel, int epoch, double loss,
			int iteration);
}