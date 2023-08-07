package main;

import scheduler_GPG.MCTS_Scheduler_GPG;
import scheduler_GPG.Match_GPG;
import scheduler_GPG.Match_Info;
import scheduler_GPG.Scheduler_GPG;
import scheduler_GPG.State_GPG;
import xml2bdi.XMLReader;
import uno.gpt.generators.*;
import util.Log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.Random;
import java.util.Scanner;

import enums.Enums.AllianceType;
import enums.Enums.ExperimentPhase;
import enums.Enums.OracleType;
import enums.Enums.RolloutEvaluationType;
import enums.Enums.VisionType;

public class Main
{
	private static final String PYTHON_EXECUTABLE = System.getProperty("user.home") + "/anaconda3/envs/aamas_23_feedback/bin/python";
	
	public static Oracle oracle;
	
	private static String log_dir_root;
	
	public static final int NUM_CONTEXTS = 3;
	public static int context;
	
	private static final int EXPERIMENT_REPETITIONS = Integer.MAX_VALUE;
	
	public static final boolean USE_LEGACY_MCTS = true;
	
	public static final boolean USE_MCTS_AS_BASELINE_SCHED = true;
	
	private static final boolean USE_UNCERTAINTY = false;
	
	private static final int NUM_BASELINE_RUNS = 0;
	
	private static final boolean KEEP_XML_FILES = false;

	private static final float MIN_TIME = 0.0f;
	private static final float MAX_TIME = 2 * 48.0f;
	
	private static final float COINS_COLLECTED_DIVISOR = 1.0f;
	private static final float ENEMIES_DEFEATED_DIVISOR = 1.0f;
	
	private static Scanner scan = new Scanner(System.in);

    public static void main(String[] args) throws Exception
    {
    	oracle = new Oracle(OracleType.HUMAN_FEEDBACK);
    	
		Log.log_to_file = false;
    	Date date = Calendar.getInstance().getTime();
        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        log_dir_root = System.getProperty("user.dir") + "/log/" + oracle.type.toString() + "_" + dateFormat.format(date) + "/";
        
		Log.refreshLogDir(log_dir_root + "baseline/", false);
		ArrayList<Float> baseline_results = run_baseline(NUM_BASELINE_RUNS);
		calculateStats(baseline_results, ExperimentPhase.BASELINE);
		
    	for (int i = 0; i < EXPERIMENT_REPETITIONS; i++)
    	{
	    	// Human experiments
	    	Log.refreshLogDir(log_dir_root + "learner/human_feedback", true);
	    	
    		// Args:
    		// - num_burn_in_runs
    		// - num_learning_runs
    		// - experiments_between_training
    		// - new_pairwise_comparisons_per_train
	    	// - noise_magnitude
	    	run_learning_experiments(10, 300, 10, 10, 0.0f);

    	}
    	
    	scan.close();
    }
    
	private static int doTraining(boolean first_train, int num_burn_in_runs, int new_training_samples,
		boolean use_uncertainty, float noise_magnitude) throws IOException, InterruptedException
	{
    	String input_file = Log.getLogDir() + "/match_results_for_python.csv";
    	String cmd = PYTHON_EXECUTABLE + " " + System.getProperty("user.dir") + "/python/train.py -n " + new_training_samples
    		+ " -v " + 0
    		+ " -b " + num_burn_in_runs
    		+ " -m " + input_file + " -o " + Log.getLogDir()
			+ " -z " + noise_magnitude;
		
    	if (!first_train)
    	{
    		cmd = cmd + " -a";
    	}
    	
    	if (use_uncertainty)
    	{
    		cmd = cmd + " -u";
    	}

		String python_log = Log.getLogDir() + "/python_log.txt";
		FileWriter fw = new FileWriter(python_log, true);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter out = new PrintWriter(bw);
        out.println(cmd);
        out.close();
        
    	Log.info("Run this training command:\n" + cmd);
    	Log.info("Then press ENTER to continue . . .\n");
        scan.nextLine();
        
        return 0;
	}
	
	private static int summariseResults() throws IOException
	{
    	String input_file = Log.getLogDir() + "/match_results_for_python.csv";
    	String cmd = PYTHON_EXECUTABLE + " " + System.getProperty("user.dir") + "/python/summarise.py " + input_file;
		
		String python_log = Log.getLogDir() + "/python_log.txt";
		FileWriter fw = new FileWriter(python_log, true);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter out = new PrintWriter(bw);
        out.println(cmd);
        out.close();
        
    	Log.info("Run this command to summarise the results:\n" + cmd);
    	Log.info("Then press ENTER to continue . . .\n");
        scan.nextLine();
        
        scan.close();
        System.exit(0);
		return 0;
	}
	
    private static void calculateStats(ArrayList<Float> raw_results, ExperimentPhase ep)
    {
		float sum_score = 0;
		for (int i = 0; i < raw_results.size(); i++)
		{
			sum_score += raw_results.get(i);
		}
		
		float mean = sum_score / raw_results.size();

		float sum_squared_deviation = 0f;
		
		for (int i = 0; i < raw_results.size(); i++)
		{
			sum_squared_deviation += Math.pow(raw_results.get(i) - mean, 2.0f);
		}
		float variance = sum_squared_deviation / (raw_results.size() - 1.0f);
		
		float std = (float)Math.sqrt(variance);

		try
		{
			String stats_file = Log.getLogDir() + "/stats.csv";
			FileWriter fw = new FileWriter(stats_file, true);
			BufferedWriter bw = new BufferedWriter(fw);
			PrintWriter out = new PrintWriter(bw);
			
			out.println("mean,std");
			out.println(mean + "," + std);
			
			out.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		
		if (ep == ExperimentPhase.BASELINE)
		{
			Oracle.baseline_mean = mean;
			Oracle.baseline_std = std;
		}
    }
    
	public static float getCompletionTimeRepresentation(float completionTimeRaw)
	{
    	float complete_time_rep = completionTimeRaw;
    	
    	if (complete_time_rep == -1f)
    	{
    		complete_time_rep = MAX_TIME;
    	}
    	complete_time_rep = (complete_time_rep - MIN_TIME) / (MAX_TIME - MIN_TIME);
    	complete_time_rep = Math.max(0.0f, Math.min(1.0f, complete_time_rep));

		return complete_time_rep;
	}
	
	public static float getCoinsRepresentation(int coinsCollectedRaw)
	{
    	float coins_collected_rep = coinsCollectedRaw / COINS_COLLECTED_DIVISOR;
		return coins_collected_rep;
	}
	
	public static float getEnemiesRepresentation(int enemiesDefeatedRaw)
	{
    	float enemies_defeated_rep = enemiesDefeatedRaw / ENEMIES_DEFEATED_DIVISOR;
		return enemies_defeated_rep;
	}
    
	public static ArrayList<Float> run_baseline(int num_baseline_runs) throws Exception
	{
		return run_experiments(num_baseline_runs, 0, 0, 0, 0, 0, 0.0f);
	}
	
	public static ArrayList<Float> run_oracle(int num_oracle_runs) throws Exception
	{
		return run_experiments(0, num_oracle_runs, 0, 0, 0, 0, 0.0f);
	}
	
	public static void run_learning_experiments(int num_burn_in_runs, int num_learning_runs, int experiments_between_training,
		int new_pairwise_comparisons_per_train, float noise_magnitude) throws Exception
	{
		run_experiments(0, 0, num_burn_in_runs, num_learning_runs, experiments_between_training, new_pairwise_comparisons_per_train, noise_magnitude);
	}
	
	public static ArrayList<Float> run_experiments(int num_baseline_runs, int num_oracle_runs, int num_burn_in_runs, int num_learning_runs,
		int experiments_between_training, int new_pairwise_comparisons_per_train, float noise_magnitude) throws Exception
	{
		ArrayList<Float> result = new ArrayList<Float>();
    	
        boolean first_train = true;
    	
    	// NOTE: If xml_file == null, a random forest will be generated
    	String xml_file = null;
    	String generatedXmlFilename = null;
    	
    	// MCTS settings
        int mcts_alpha = 100;
        int mcts_beta = 10;
        
        double mcts_c = Math.sqrt(2.0);
        
        // GPG Schedulers
		ArrayList<String> gpg_scheduler_names = new ArrayList<String>();
		ArrayList<Scheduler_GPG> gpg_schedulers = new ArrayList<Scheduler_GPG>();
		
		gpg_scheduler_names.add("Baseline_scheduler");
		if (USE_MCTS_AS_BASELINE_SCHED)
		{
			gpg_schedulers.add(new MCTS_Scheduler_GPG(VisionType.FULL, mcts_alpha, mcts_beta, mcts_c, 1.0, null, RolloutEvaluationType.DEFAULT, USE_LEGACY_MCTS, true));
		}

		gpg_scheduler_names.add("UNUSED");
		gpg_schedulers.add(null);

		gpg_scheduler_names.add("MCTS_learned_eval");
		gpg_schedulers.add(new MCTS_Scheduler_GPG(VisionType.FULL, mcts_alpha, mcts_beta, mcts_c, 1.0, null, RolloutEvaluationType.LEARNED, USE_LEGACY_MCTS, true));

		// GPT generator settings
    	int depth = 3;
    	int numEnvironmentVariables = 30;
    	int numVarToUseAsPostCond = 15;
    	int numGoalPlanTrees = 4;
    	int subgoalsPerPlan = 1;
    	int plansPerGoal = 2;
    	int actionsPerPlan = 3;
    	double propGuaranteedCPrecond = 0.5;

    	XMLReader reader;
        Random rm = new Random();
    	
        for (int experiment_num = 0; experiment_num < (num_baseline_runs + num_oracle_runs + num_burn_in_runs + num_learning_runs); experiment_num++)
        {
        	ExperimentPhase experiment_phase;
        	if (experiment_num < num_baseline_runs)
        	{
        		experiment_phase = ExperimentPhase.BASELINE;
        	}
        	else if (experiment_num < (num_baseline_runs + num_oracle_runs + num_burn_in_runs))
        	{
        		experiment_phase = ExperimentPhase.BURN_IN;
        	}
        	else
        	{
        		experiment_phase = ExperimentPhase.LEARNING;
        	}

        	// Train the neural net if applicable
        	if (experiment_phase == ExperimentPhase.LEARNING)
        	{
        		int runs_since_learning_commenced = experiment_num - num_baseline_runs - num_oracle_runs - num_burn_in_runs;
        		
        		if (runs_since_learning_commenced % experiments_between_training == 0)
        		{
        			if (experiment_num == 30)
        			{
        				summariseResults();
        			}
        			else
        			{
                		doTraining(first_train, num_burn_in_runs, new_pairwise_comparisons_per_train, USE_UNCERTAINTY, noise_magnitude);
                		first_train = false;
        			}
        		}
        	}
        	
	    	String generatorArgsStr = "";
	    	
        	String generated_filename = "random_" + experiment_num + ".xml";
        	
	    	if (xml_file == null)
	    	{
		    	int randomSeed = rm.nextInt();
		    	
		        generatedXmlFilename = Log.getLogDir() + "/" + generated_filename;
		    	
		    	String[] generatorArgs = new String[21];
		    	generatorArgs[0] = "synth";
		    	generatorArgs[1] = "-f";
		    	generatorArgs[2] = generatedXmlFilename;
		    	generatorArgs[3] = "-s";
		    	generatorArgs[4] = Integer.toString(randomSeed);
		    	generatorArgs[5] = "-d";
		    	generatorArgs[6] = Integer.toString(depth);
		    	generatorArgs[7] = "-t";
		    	generatorArgs[8] = Integer.toString(numGoalPlanTrees);
		    	generatorArgs[9] = "-v";
		    	generatorArgs[10] = Integer.toString(numEnvironmentVariables);
		    	generatorArgs[11] = "-g";
		    	generatorArgs[12] = Integer.toString(subgoalsPerPlan);
		    	generatorArgs[13] = "-p";
		    	generatorArgs[14] = Integer.toString(plansPerGoal);
		    	generatorArgs[15] = "-a";
		    	generatorArgs[16] = Integer.toString(actionsPerPlan);
		    	generatorArgs[17] = "-y";
		    	generatorArgs[18] = Double.toString(propGuaranteedCPrecond);
		    	generatorArgs[19] = "-w";
		    	generatorArgs[20] = Integer.toString(numVarToUseAsPostCond);
		    	
		    	for (int argNum = 0; argNum < generatorArgs.length; argNum++)
		    	{
		    		generatorArgsStr = generatorArgsStr + generatorArgs[argNum] + " ";
		    	}
		    	Log.info(generatorArgsStr);

		    	XMLGenerator.generate(generatorArgs);
		    	
		    	reader = new XMLReader(generatedXmlFilename);
	    	}
	    	else
	    	{
	            reader = new XMLReader(xml_file);
	    	}
	    	
	    	String forest_name = (xml_file == null)? generated_filename : xml_file;
	    	
	        // Read the initial state from the XML file
	    	State_GPG currentStateGPG = new State_GPG(forest_name, reader.getBeliefs(), reader.getIntentions(), reader.getNodeLib(), reader.getPreqMap(), reader.getParentMap(), 0);
	    	
	    	currentStateGPG.reset_linearisations();
	    	
	    	// Use random agent for the burn in experiments
	    	int agent_1;
	    	
	    	switch (experiment_phase)
	    	{
		    	case BASELINE:
		    	case BURN_IN:
		    		agent_1 = 0;
		    		break;
		    	case LEARNING:
		    		agent_1 = 2;
		    		break;
	    		default:
	    	    	Log.info("ERROR: Unhandled experiment phase (" + experiment_phase.toString() + ")");
	    	    	agent_1 = -1;
	    			System.exit(0);
	    	}

        	if (gpg_schedulers.get(agent_1) instanceof MCTS_Scheduler_GPG)
        	{
        		((MCTS_Scheduler_GPG)gpg_schedulers.get(agent_1)).payoff_matrix = new double[][] {{1.0}};
        	}

        	context = rm.nextInt(NUM_CONTEXTS);
        	
        	Match_Info mi = new Match_GPG("solo_test_" + gpg_scheduler_names.get(agent_1),
	            	numGoalPlanTrees,
	            	AllianceType.ADVERSARIAL,
	            	currentStateGPG.clone(),
	            	new Scheduler_GPG[] {gpg_schedulers.get(agent_1)},
	            	new String[] {gpg_scheduler_names.get(agent_1)}).run(true, true, false);
	        
        	if (experiment_phase == ExperimentPhase.BASELINE)
        	{
        		result.add(oracle.getScore(mi.final_state, false));
        	}
	        
	        if (!KEEP_XML_FILES)
	        {
	        	File xmlFile = new File(generatedXmlFilename); 
	        	xmlFile.delete();
	        }
        }
        
        return result;
	}
}