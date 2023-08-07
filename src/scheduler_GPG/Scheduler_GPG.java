package scheduler_GPG;

public abstract class Scheduler_GPG {
	
	public Match_GPG match;
	public int agent_num;
	public boolean mirror_match;
	
	// The *true* agent objectives that are used for calculating scores in the results.
	// This is different from the alliance matrices used by the MCTS agents, in that the latter
	// capture the agents' *assumed* objectives.
	public double[][] agentObjectives;
	
	public boolean[][] available_intentions;
	public double[][] intention_values;
	
    public abstract Decision_GPG getDecision(State_GPG state);
    
    public abstract void reset();
    
    public void loadMatchDetails(Match_GPG match, int agent_num, boolean mirror_match)
    {
    	this.match = match;
    	this.agent_num = agent_num;
    	this.mirror_match = mirror_match;
    	
		this.agentObjectives = new double[match.numAgents][match.numAgents];
		this.available_intentions = new boolean[match.schedulers.length][match.numGoalPlanTrees];
		this.intention_values = new double[match.schedulers.length][match.numGoalPlanTrees];
		
		if (match.allianceType == match.allianceType.TWO_VS_TWO)
		{
			agentObjectives[0] = new double[] {1.0, -1.0, 1.0, -1.0};
			agentObjectives[1] = new double[] {-1.0, 1.0, -1.0, 1.0};
			agentObjectives[2] = new double[] {1.0, -1.0, 1.0, -1.0};
			agentObjectives[3] = new double[] {-1.0, 1.0, -1.0, 1.0};
		}
		else if (match.allianceType == match.allianceType.P1_HELPER)
		{
			agentObjectives[0] = new double[] {0.0, 1.0, 1.0};
			agentObjectives[1] = new double[] {0.0, 1.0, 0.0};
			agentObjectives[2] = new double[] {0.0, 0.0, 1.0};
		}
		else if (match.allianceType == match.allianceType.P1_MEDDLER)
		{
			agentObjectives[0] = new double[] {0.0, 1.0, -1.0};
			agentObjectives[1] = new double[] {0.0, 1.0, 0.0};
			agentObjectives[2] = new double[] {0.0, 0.0, 1.0};
		}
		else
		{
			for (int i = 0; i < match.numAgents; i++)
			{
				for (int j = 0; j < match.numAgents; j++)
				{
					if (i == j)
					{
						// An agent is always allied with itself
						agentObjectives[i][j] = 1.0;
					}
					else
					{
						switch(match.allianceType)
						{
							case ADVERSARIAL:
								agentObjectives[i][j] = -1.0;
								break;
								
							case ALLIED:
								agentObjectives[i][j] = 1.0;
								break;
								
							case NEUTRAL:
							default:
								agentObjectives[i][j] = 0.0;
						}
					}
				}
			}
		}
		
		for (int intentionNum = 0; intentionNum < match.numGoalPlanTrees; intentionNum++)
		{
			int agentToAssignIntention = mirror_match? ((intentionNum + 1) % match.numAgents) : (intentionNum % match.numAgents);
			
			for (int agentNum = 0; agentNum < match.numAgents; agentNum++)
			{
				available_intentions[agentNum][intentionNum] = (agentNum == agentToAssignIntention);
			}
			
			for (int agentNum = 0; agentNum < match.numAgents; agentNum++)
			{
				intention_values[agentNum][intentionNum] = agentObjectives[agentNum][agentToAssignIntention];
			}
		}
    }
    
    public int getAssignedAgent(int intention_num)
    {
		for (int agentNum = 0; agentNum < match.numAgents; agentNum++)
		{
			if (available_intentions[agentNum][intention_num])
			{
				return agentNum;
			}
		}
		
		return -1;
    }
}
