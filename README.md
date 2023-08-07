# Feedback-Guided Intention Scheduling for BDI Agents

Source code for the AAMAS-23 paper *Feedback-Guided Intention Scheduling for BDI Agents*, by Michael Dann, John Thangarajah and Minyi Li.

## Running

This project has glued together from several different pieces and remains very much "research code". Apologies in advance!

Most of the core logic is written in Java (since we're building on previous work that was implemented in Java), but the machine learning aspects are handled via Python. To install the Python requirements via Anaconda, use
```conda env create -f aamas_23_feedback.yml```.

The Java code is set up to run the necessary Python scripts automatically, but you'll need to ensure that the path to the python executable is set up correctly in Main.java. The default is:<br />
```private static final String PYTHON_EXECUTABLE = System.getProperty("user.home") + "/anaconda3/envs/aamas_23_feedback/bin/python";```

## Running the Oracle Experiments

The code in Main.java is set up to reproduce the results from the paper, but it takes quite a while to run, so be warned!

The experiment type is controlled via arg[0].
* Use WEIGHTED_GOALS to generate the results from Figure 2(a) in the paper.
* Use CONTEXTUAL_WEIGHTED_GOALS to generate the results from Figure 2(b) in the paper.
* Use TIME_TAKEN to generate the results from Figure 2(c) in the paper.
* Use TIME_TAKEN_NOISY to generate the results from Figure 3 in the paper.

The code also includes options for noisy versions of the weighted goals oracles: WEIGHTED_GOALS_NOISY and CONTEXTUAL_WEIGHTED_GOALS_NOISY.

The first thing the code does is to calculate the results for MCTS-oracle and MCTS-goals, i.e., the dotted lines in Figures 2 and 3. You can skip over this by setting NUM_ORACLE_RUNS = 0 and NUM_BASELINE_RUNS = 0 in Main.java.

Next, the code generates the results for our approach, pref-MCTS, using the three feedback frequencies considered in the paper: N<sub>c</sub> = 100, N<sub>c</sub> = 25 and N<sub>c</sub> = 5.

Results are automatically saved to the log folder.

## Running the User Study

* Switch to the human_experiments branch of this repo.
* Run Main.java (there's no need to specify any command-line arguments).
* Initially code will automatically generate 10 "default" trajectories for the human user to provide preferences over.
* Once this completes, the Java code will output a console command to run. Run this in a separate terminal window and provide preferences however you see fit. (An easy way to verify that the approach is learning is to always select the trajectory with more coins collected.)
* Once the training completes, come back to the Java program and press ENTER.
* The scheduler will now generate 10 more trajectories, using the provided preference information to (hopefully) improve the quality of the trajectories.
* After the second set of trajectories completes, the Java code will again output a console command to run. Run it and provide preferences as before, then return to the Java program and press ENTER.
* The scheduler will now generate a final set of 10 trajectories, taking all of the feedback so far into account.
* After this last set of trajectories completes, the Java code will output a final console command to run that summarises the results.
