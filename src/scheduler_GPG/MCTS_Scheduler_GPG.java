package scheduler_GPG;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import main.Main;
import enums.Enums.RolloutEvaluationType;
import enums.Enums.VisionType;
import goalplantree.ActionNode;
import goalplantree.TreeNode;
import nn.FeedForwardNetwork;
import util.Log;

public class MCTS_Scheduler_GPG extends Scheduler_GPG {

	public VisionType vision_type;
	
	public MCTS_Node_GPG rootNode;
	
	public int alpha, beta;
	public double c, rollout_stochasticity;
	public double[][] payoff_matrix;
	public RolloutEvaluationType rollout_eval_type;
	public Scheduler_GPG rollout_schedulers[];
	public boolean[] gpt_visible;
	private boolean legacy;
	private boolean build_tree;
	private double best_rollout_return;
	private State_GPG best_rollout_end_state;
	private FeedForwardNetwork ffn;
	
    // random
    static Random rm = new Random();
    
    // a very small value used for breaking the tie and dividing by 0
    static final double epsilon = 1e-6;
	
    // statistics
    public int nRollouts;
    
    public MCTS_Scheduler_GPG(VisionType vision_type, int alpha, int beta, double c, double rollout_stochasticity,
    	double[][] payoff_matrix, RolloutEvaluationType rollout_eval_type, boolean legacy, boolean build_tree)
    {
    	this.vision_type = vision_type;
    	this.alpha = alpha;
    	this.beta = beta;
    	this.c = c;
    	this.rollout_stochasticity = rollout_stochasticity;
    	this.payoff_matrix = payoff_matrix;
    	this.rollout_eval_type = rollout_eval_type;
    	this.legacy = legacy;
    	this.build_tree = build_tree;
    	this.best_rollout_return = Double.NEGATIVE_INFINITY;
    	this.best_rollout_end_state = null;
    }
    
	@Override
	public void reset()
	{
		if (rollout_eval_type == RolloutEvaluationType.LEARNED)
		{
			this.ffn = new FeedForwardNetwork(Log.getLogDir());
		}
		
		this.best_rollout_return = Double.NEGATIVE_INFINITY;
    	this.best_rollout_end_state = null;
	}
    
    @Override
    public void loadMatchDetails(Match_GPG match, int agent_num, boolean mirror_match)
    {
    	super.loadMatchDetails(match, agent_num, mirror_match);
    	
		this.gpt_visible = new boolean[match.numGoalPlanTrees];
		
		for (int intentionNum = 0; intentionNum < match.numGoalPlanTrees; intentionNum++)
		{
			int agentToAssignIntention = mirror_match? ((intentionNum + 1) % match.numAgents) : (intentionNum % match.numAgents);
			
			if (agentToAssignIntention == agent_num)
			{
				gpt_visible[intentionNum] = true; // Can always see own GPTs
			}
			else
			{
				switch(vision_type)
				{
					case FULL:
						gpt_visible[intentionNum] = true;
						break;
						
					case PARTIALLY_AWARE:
						
						if ((intentionNum / match.numAgents) % 2 == 0)
						{
							gpt_visible[intentionNum] = true;
						}
						else
						{
							gpt_visible[intentionNum] = false;
						}

						break;
						
					case UNAWARE:
					default:
						gpt_visible[intentionNum] = false;
				}
			}
		}
		
    	this.rollout_schedulers = new Scheduler_GPG[match.numAgents];
    	for (int i = 0; i < match.numAgents; i++)
    	{
            this.rollout_schedulers[i] = new Random_Scheduler_GPG(legacy, true);
    	}
    }
    
    public Decision_GPG getDecision(State_GPG state)
    {
    	if (state.isGameOver())
    	{
    		rootNode = null; // Free memory
    		match = null;
    		System.gc();
    		return new Decision_GPG(null, true);
    	}
    	
    	// Check if the only action available is pass
    	ArrayList<TreeNode> expansionActions = state.getExpansionActions(available_intentions[state.playerTurn], legacy, match.agent_names[agent_num]);
    	
    	boolean playerMustPass = (((expansionActions.size() == 1) && (expansionActions.get(0) == null)))
    		|| (expansionActions.size() == 0);
    	
    	if (playerMustPass)
    	{
    		rootNode = null; // Free memory
    		//match = null;
    		System.gc();
    		return new Decision_GPG(null, true);
    	}
    	
    	rootNode = new MCTS_Node_GPG(state, match);
    	nRollouts = 0;
    	
    	run(alpha, beta);
    	
    	ArrayList<TreeNode> actionChoices = new ArrayList<TreeNode>();
    	
        MCTS_Node_GPG currentNode = rootNode;
        
        boolean first_action_in_chain = true;
        
        while (true)
	    {
            TreeNode actionChoice = null;
        	int selected_idx = -1;
	        int visits = -1;
	        double selected_best_val = Double.NEGATIVE_INFINITY;
	        double selected_average = Double.NEGATIVE_INFINITY;

	        Log.info("\nDepth " + actionChoices.size() + " actions available:");
	        
	        for (int i = 0; i < currentNode.children.size(); i++)
	        {
	        	TreeNode tmp_action_choice = currentNode.children.get(i).getActionChoice();
	        	
	        	if (tmp_action_choice == null)
	        	{
	            	Log.info("PASS"
	    	    	+ ": Best val = " + (currentNode.children.get(i).bestValue[agent_num])
	    			+ ", Ave. val = " + (currentNode.children.get(i).totValue[agent_num] / currentNode.children.get(i).nVisits)
	    			+ ", visits = " + currentNode.children.get(i).nVisits);
	        	}
	        	else
	        	{
	            	Log.info("Action " + tmp_action_choice.getType()
	            	+ ": Best val = " + (currentNode.children.get(i).bestValue[agent_num])
	    			+ ", Ave. val = " + (currentNode.children.get(i).totValue[agent_num] / currentNode.children.get(i).nVisits)
	    			+ ", visits = " + currentNode.children.get(i).nVisits);
	        	}
	
	        	boolean update_selection = false;
	        	
	        	// Make sure we don't, for example, select a plan node and *then* select PASS.
	        	if ((currentNode.children.get(i).getActionChoice() != null) || first_action_in_chain || (currentNode.children.size() == 1))
	        	{
	        		double node_ave;
	        		if (currentNode.children.get(i).nVisits == 0)
	        		{
	        			node_ave = Double.NEGATIVE_INFINITY + 1.0;
	        		}
	        		else
	        		{
	        			node_ave = currentNode.children.get(i).totValue[agent_num] / currentNode.children.get(i).nVisits;
	        		}

			        update_selection = node_ave > selected_average;
	        	}

	            if (update_selection)
	            {
	            	selected_idx = i;
	            	actionChoice = currentNode.children.get(i).getActionChoice();
	                visits = currentNode.children.get(i).nVisits;
	                selected_best_val = currentNode.children.get(i).bestValue[agent_num];
	                
	                if (currentNode.children.get(i).nVisits == 0)
	                {
	                	selected_average = Double.NEGATIVE_INFINITY + 1.0;
	                }
	                else
	                {
	                	selected_average = currentNode.children.get(i).totValue[agent_num] / currentNode.children.get(i).nVisits;
	                }
	            }
	        }
	        
	    	actionChoices.add(actionChoice);
	    	
	    	if (actionChoice == null)
	    	{
	    		Log.info("Action choice: PASS"
	    			+ " (Best val " + selected_best_val
	    			+ ", average " + selected_average + " from " + visits + " visits)");
	    	}
	    	else
	    	{
	    		Log.info("Action choice: " + actionChoice.getType()
	    	    	+ " (Best val " + selected_best_val
	    	    	+ ", average " + selected_average + " from " + visits + " visits)");
	    	}

	    	currentNode = currentNode.children.get(selected_idx);
	    	
	    	if (actionChoice == null || actionChoice instanceof ActionNode)
	    	{
	    		break;
	    	}
	    		
	    	// In some situations, traversing the tree by following the greedy branch doesn't actually
	    	// lead to an action node. In this case, expand below until we find an action node.
	    	if (currentNode.children == null)
	    	{
            	boolean[] intentionAvailable = new boolean[match.numGoalPlanTrees];
            	for (int int_num = 0; int_num < match.numGoalPlanTrees; int_num++)
            	{
            		intentionAvailable[int_num] = available_intentions[currentNode.state.playerTurn][int_num] && gpt_visible[int_num];
            	}
            	
            	if (!currentNode.expanded)
            	{
            		currentNode.expand(intentionAvailable, legacy, match.agent_names[agent_num]);
            	}
            	
	    		double selected_mean_value = Double.NEGATIVE_INFINITY;
	    		TreeNode best_action_choice = null;
	    		
	    		for (MCTS_Node_GPG c : currentNode.children)
	    		{
	    			TreeNode node_choice = c.getActionChoice();
	    			
	    			if (node_choice instanceof ActionNode)
	    			{
	    				double mean_value = Double.NEGATIVE_INFINITY + 1.0;
	    				
	    				if (currentNode.nVisitsAllNodes.containsKey(node_choice))
	    				{
	    					mean_value = currentNode.totValueAllNodes.get(node_choice)[currentNode.state.playerTurn] / currentNode.nVisitsAllNodes.get(node_choice);
	    				}
	    				
    					if (mean_value > selected_mean_value)
    					{
    						selected_mean_value = mean_value;
    						best_action_choice = node_choice;
    					}
	    			}
	    		}
	    		
	    		if (selected_mean_value > Double.NEGATIVE_INFINITY)
	    		{
		    		Log.info("\nAction choice: " + best_action_choice.getType()
	        		+ " (Rollout average of " + selected_mean_value + ")");
		    		
    				actionChoices.add(best_action_choice);
	    			break;
	    		}
	    		else
	    		{
	    			Log.info("ERROR: MCTS failed to select an action!", true);
	    		}
	    	}
	    	
	    	first_action_in_chain = false;
	    }
 
        Log.info("\nBest rollout return from all sims so far: " + best_rollout_return);
        
        Decision_GPG result = new Decision_GPG(actionChoices, (rootNode.children.size() == 1) && (rootNode.children.get(0).actionChoice == null));
        System.gc();
        rootNode = null; // Free memory
        
        return result;
    }
    
    /**
     * @return a node with maximum UCT value
     */
    private MCTS_Node_GPG select(MCTS_Node_GPG currentNode, boolean forceAction)
    {
        // Initialisation
    	MCTS_Node_GPG selected = null;
        double bestUCT = Double.NEGATIVE_INFINITY;
        
        // Calculate the UCT value for each of its selected nodes
        for(int i = 0; i < currentNode.children.size(); i++)
        {
            // UCT calculation
            double uctValue = currentNode.children.get(i).totValue[currentNode.state.playerTurn] / (currentNode.children.get(i).nVisits + epsilon)
            	+ c * Math.sqrt(Math.log(nRollouts + 1) / (currentNode.children.get(i).nVisits + epsilon))
            	+ epsilon * rm.nextDouble(); // For tie-breaking
            
            // Compare with the current maximum value
            if (uctValue > bestUCT)
            {
                selected = currentNode.children.get(i);
                bestUCT = uctValue;
            }
        }
        
        // Return the nodes with maximum UCT value, null if current node is a leaf node (contains no child nodes)
        return selected;
    }
    
    /**
     * The main MCTS process
     * @param alpha number of iterations
     * @param beta number of simulation per iteration
     */
    private void run(int alpha, int beta)
    {	 
        long startTime = System.currentTimeMillis();
        
        // Record the list of nodes that has been visited
        List<MCTS_Node_GPG> visited = new LinkedList<>();
        
        // Run alpha iterations
        for(int i = 0; i < alpha; i++)
        {
        	//Log.info("MCTS iter: " + (i + 1) + " / " + alpha, true);
        	
            visited.clear();
            
            // Set the current node to this node
            MCTS_Node_GPG currentNode = this.rootNode;
            
            // Add this node to the list of visited node
            visited.add(currentNode);
            
            //if (currentNode.getActionChoice() != null)
            //{
            //	Log.info("MCTS selected: " + currentNode.getActionChoice().getType(), true);
            //}
            //else
            //{
            //	Log.info("MCTS selected PASS", true);
            //}
            
            // Find the leaf node which has the largest UCT value
            //Log.info("Beginning selection...");
            
            if (build_tree)
            {
	            while ((currentNode != null) && !currentNode.isLeaf())
	            {
	            	//MCTS_Node_GPG tmpNode = currentNode;
	                currentNode = select(currentNode, false);
	                
	                if (currentNode != null)
	                {
	                    visited.add(currentNode);
	                    
	                    if (currentNode.getActionChoice() != null)
	                    {
	                    	//Log.info("MCTS selected: " + currentNode.getActionChoice().getType() + " for player " + tmpNode.state.playerTurn, true);
	                    }
	                    else
	                    {
	                    	//Log.info("MCTS selected PASS" + " for player " + tmpNode.state.playerTurn, true);
	                    }
	                }
	            }
	            //Log.info("Ended selection.\n");
            }
            
            if (!currentNode.state.isGameOver())
            {
            	boolean[] intentionAvailable = new boolean[match.numGoalPlanTrees];
            	for (int int_num = 0; int_num < match.numGoalPlanTrees; int_num++)
            	{
            		intentionAvailable[int_num] = available_intentions[currentNode.state.playerTurn][int_num] && gpt_visible[int_num];
            	}
            	
            	if (!currentNode.expanded)
            	{
            		currentNode.expand(intentionAvailable, legacy, match.agent_names[agent_num]);
            	}
            	
	            // Select a node for simulation
	            currentNode = select(currentNode, true);
	            visited.add(currentNode);
	            
	            if (currentNode.getActionChoice() != null)
	            {
	            	//Log.info("MCTS selected: " + currentNode.getActionChoice().getType(), true);
	            	//Log.info("Selected action STILL valid? " + currentNode.state.isCandidate(currentNode.getActionChoice()), true);
	            }
	            else
	            {
	            	//Log.info("MCTS selected PASS", true);
	            }
            }
            
            // Simulation
            for (int j = 0; j < beta; j++)
            {
            	State_GPG endOfGame;
            	HashSet<TreeNode> nodes_visited = new HashSet<TreeNode>();
            	
            	if (currentNode.state.isGameOver())
            	{
            		endOfGame = currentNode.state;
            	}
            	else
            	{
	            	//Log.info("MCTS sub iter: " + (j + 1) + " / " + beta, true);
	            	
	            	// TODO: This assumes that the rollout schedulers are not stateful. It might be safer to clone them.
            		// NOTE: Need to use *this* scheduler's name for the rollout schedulers so that they see the same linearisation. Should probably make the code less convoluted!
            		
	                //Match_GPG m = new Match_GPG("MCTS_rollout", match.numGoalPlanTrees, match.allianceType, currentNode.state.clone(),
		            //    	rollout_schedulers, new String[] {match.agent_names[agent_num], match.agent_names[agent_num]}, match.assumed_politeness);
	                
            		String[] sim_agent_names = new String[match.numAgents];
                	for (int k = 0; k < match.numAgents; k++)
                	{
                		sim_agent_names[k] = match.agent_names[agent_num];
                	}
            		
	                Match_GPG m = new Match_GPG("MCTS_rollout", match.numGoalPlanTrees, match.allianceType, currentNode.state.clone(),
	                	rollout_schedulers, sim_agent_names);
	                
	                Match_Info m_info = m.run(false, false, mirror_match, gpt_visible);
	                
	                endOfGame = m_info.final_state;
	                nodes_visited = m_info.nodes_visited;
            	}
                
                nRollouts++;
		        
    	        double[] agent_scores = new double[match.numAgents];
    	        
    	        switch (rollout_eval_type)
    	        {
	    	        case DEFAULT:
	    	        	for (int intNum = 0; intNum < endOfGame.intentions.size(); intNum++)
		    	        {
		    	        	if (gpt_visible[intNum])
		    	        	{
		        	            if (endOfGame.isIntentionComplete(intNum))
		    	                {
		        	            	int assigned_agent = getAssignedAgent(intNum);
		        	            	
			                		for (int k = 0; k < match.numAgents; k++)
			                		{
			                			agent_scores[k] += payoff_matrix[k][assigned_agent];
			                		}
		    	                }
		    	        	}
		    	        }
	    	        	break;
	    	        	
	    	        //case ORACLE:
	    	        //	agent_scores[0] = Main.oracle.getScore(endOfGame, true);
	    	        //	break;
	    	        	
	    	        case LEARNED:
	    	        	
	                    ////////////////////////////////
	                    // Determine neural net input //
	                    ////////////////////////////////
	                    int nn_input_size = endOfGame.intentions.size();

	    		        if (Main.oracle.uses_time)
	                    {
	    		        	nn_input_size += endOfGame.intentions.size();
	                    }
	    		        
	    		        if (Main.oracle.uses_coins)
	                    {
	    		        	nn_input_size += 1;
	                    }
	    		        
	    		        if (Main.oracle.uses_enemies)
	                    {
	    		        	nn_input_size += 1;
	                    }
	    		        
	                    float[] nn_input = new float[nn_input_size];
	                    
	                    for (int intNum = 0; intNum < endOfGame.intentions.size(); intNum++)
	        	        {
	        	        	if (gpt_visible[intNum])
	        	        	{
	        	        		nn_input[intNum] = endOfGame.isIntentionComplete(intNum) ? 1.0f : 0.0f;
	        	        	}
	        	        	
	                        if (Main.oracle.uses_time)
	                        {
	        	        		nn_input[intNum + endOfGame.intentions.size()] = Main.getCompletionTimeRepresentation(endOfGame.intention_completion_times[intNum]);
	                        }
	        	        }
	                    
	    		        if (Main.oracle.uses_coins)
	                    {
	    		        	int offset = Main.oracle.uses_enemies ? 2 : 1;
	    			        nn_input[nn_input_size - offset] = Main.getCoinsRepresentation(endOfGame.coins_collected);
	                    }
	    		        
	    		        if (Main.oracle.uses_enemies)
	                    {
	    			        nn_input[nn_input_size - 1] = Main.getEnemiesRepresentation(endOfGame.enemies_defeated);
	                    }
	                    ////////////////////////////////
	    		        
	    	        	agent_scores[0] = ffn.CalculateOutput(nn_input)[0];
	    	        	break;
	    	        	
	        		default:
	        			Log.info("ERROR: Unhandled rollout evaluation type (" + rollout_eval_type.toString() + ")");
	        			System.exit(0);
    	        }
    	        
    	        if (agent_scores[agent_num] > best_rollout_return)
    	        {
    	        	best_rollout_return = agent_scores[agent_num];
    	        	best_rollout_end_state = endOfGame.clone();
    	        }
    	        
                // Back-propagation
                for(MCTS_Node_GPG node : visited)
                {
                    node.nVisits++;

    				for (int k = 0; k < match.numAgents; k++)
    				{
    					if (agent_scores[k] > node.bestValue[k])
    					{
    						node.bestValue[k] = agent_scores[k];
    					}
    					
    	                node.totValue[k] += agent_scores[k];
    	        		node.totSqValue[k] += agent_scores[k] * agent_scores[k];
    				}
        			
                    for (TreeNode t : nodes_visited)
                    {
                    	int new_visits = 1;
                    	double[] newTotValue = new double[match.schedulers.length];
                    	if (node.nVisitsAllNodes.containsKey(t))
                    	{
                    		new_visits = node.nVisitsAllNodes.get(t) + 1;
                    		newTotValue = node.totValueAllNodes.get(t);
                    	}
                    	
                    	double[] newBestValue = new double[match.schedulers.length];
                    	if (node.bestValueAllNodes.containsKey(t))
                    	{
                    		newBestValue = node.bestValueAllNodes.get(t);
                    	}
                    	
                    	node.nVisitsAllNodes.put(t, new_visits);
                    	
                		for (int k = 0; k < match.numAgents; k++)
                		{
                			newTotValue[k] += agent_scores[k];
                			
        					if (agent_scores[k] > newBestValue[k])
        					{
        						newBestValue[k] = agent_scores[k];
        					}
                		}
            			
            			node.totValueAllNodes.put(t, newTotValue);
            			node.bestValueAllNodes.put(t, newBestValue);
                    }
                }
            }
        }

        Log.info("MCTS calculation time = " + (System.currentTimeMillis() - startTime) + "ms");
    }
}
