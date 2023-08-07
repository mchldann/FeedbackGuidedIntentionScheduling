package main;

import enums.Enums.OracleType;
import scheduler_GPG.State_GPG;
import util.Log;

public class Oracle
{
	public static float baseline_mean = 0.0f;
	public static float baseline_std = 0.0f;
	
	public OracleType type;
	public boolean noisy;
	public boolean uses_time;
	public boolean uses_context;
	
    public Oracle(OracleType type)
    {
        this.type = type;
        
    	switch (type)
    	{
    		case WEIGHTED_GOALS:
    			noisy = false;
    			uses_time = false;
    			uses_context = false;
    			break;
    			
    		case WEIGHTED_GOALS_NOISY:
    			noisy = true;
    			uses_time = false;
    			uses_context = false;
    			break;
    			
    		case CONTEXTUAL_WEIGHTED_GOALS:
    			noisy = false;
    			uses_time = false;
    			uses_context = true;
    			break;
    			
    		case CONTEXTUAL_WEIGHTED_GOALS_NOISY:
    			noisy = true;
    			uses_time = false;
    			uses_context = true;
    			break;
    			
    		case TIME_TAKEN:
    			noisy = false;
    			uses_time = true;
    			uses_context = false;
    			break;
    			
    		case TIME_TAKEN_NOISY:
    			noisy = true;
    			uses_time = true;
    			uses_context = false;
    			break;
    			
    		default:
    			System.out.println("ERROR: Unhandled oracle type (" + type.toString() + ")");
    			System.exit(0);
    	}
    }
    
    public float getScore(State_GPG final_state, boolean normalise)
    {
    	switch (type)
    	{
    		case WEIGHTED_GOALS:
    		case WEIGHTED_GOALS_NOISY:
    			return getScoreGoalsOnly(final_state, normalise);
    		case CONTEXTUAL_WEIGHTED_GOALS:
    		case CONTEXTUAL_WEIGHTED_GOALS_NOISY:
    			return getScoreGoalsOnlyWithContext(final_state, normalise);
    		case TIME_TAKEN:
    		case TIME_TAKEN_NOISY:
    			return getScoreTimeElapsed(final_state, normalise);
    		default:
    	    	Log.info("ERROR: Unhandled oracle type (" + type.toString() + ")");
    			System.exit(0);
    			return Float.NEGATIVE_INFINITY;
    	}
    }
    
    private float getScoreTimeElapsed(State_GPG final_state, boolean normalise)
    {
    	float[] goal_vals = new float[10];

	    goal_vals[0] = 0.0f;
	    goal_vals[1] = 0.0f;
	    goal_vals[2] = 0.0f;
	    goal_vals[3] = 0.0f;
	    goal_vals[4] = 0.2f;
	    goal_vals[5] = 0.4f;
	    goal_vals[6] = 0.6f;
	    goal_vals[7] = 0.8f;
	    goal_vals[8] = 1.0f;
	    goal_vals[9] = 2.0f;

	    float score = 0.0f;

	    for (int i = 0; i < final_state.intentions.size(); i++)
	    {
	        score -= goal_vals[i] * Main.getCompletionTimeRepresentation(final_state.intention_completion_times[i]);
	    }
	    
	    // Non-linear terms
	    score -= 4.0 * Math.pow(Main.getCompletionTimeRepresentation(final_state.intention_completion_times[0]), 2.0);
	    
	    score -= (Main.getCompletionTimeRepresentation(final_state.intention_completion_times[1]) < 0.5f) ? 0.0 : 3.0;
	    
	    score -= 4.0 * Main.getCompletionTimeRepresentation(final_state.intention_completion_times[2]) * Main.getCompletionTimeRepresentation(final_state.intention_completion_times[3]);
	    
	    if (normalise)
	    {
	    	score = (score - baseline_mean) / baseline_std;
	    }
	    
	    return score;
    }

    private float getScoreGoalsOnly(State_GPG final_state, boolean normalise)
    {
    	float[] goal_vals = new float[10];

	    goal_vals[0] = 0.5f;
	    goal_vals[1] = 0.5f;
	    goal_vals[2] = 0.5f;
	    goal_vals[3] = 0.5f;
	    goal_vals[4] = 1.0f;
	    goal_vals[5] = 1.0f;
	    goal_vals[6] = 1.0f;
	    goal_vals[7] = 2.0f;
	    goal_vals[8] = 2.0f;
	    goal_vals[9] = 4.0f;
	    
	    float score = 0.0f;
	    
	    for (int i = 0; i < final_state.intentions.size(); i++)
	    {
	        score += goal_vals[i] * (final_state.isIntentionComplete(i) ? 1 : 0);
	    }
	    
	    if (normalise)
	    {
	    	score = (score - baseline_mean) / baseline_std;
	    }
	    
	    return score;
    }
    
    private float getScoreGoalsOnlyWithContext(State_GPG final_state, boolean normalise)
    {
    	float[] goal_vals = new float[10];

    	switch (Main.context)
    	{
	    	case 0:
			    goal_vals[0] = 0.5f;
			    goal_vals[1] = 0.5f;
			    goal_vals[2] = 0.5f;
			    goal_vals[3] = 0.5f;
			    goal_vals[4] = 1.0f;
			    goal_vals[5] = 1.0f;
			    goal_vals[6] = 1.0f;
			    goal_vals[7] = 2.0f;
			    goal_vals[8] = 2.0f;
			    goal_vals[9] = 4.0f;
			    break;
	    	case 1:
			    goal_vals[0] = 1.0f;
			    goal_vals[1] = 2.0f;
			    goal_vals[2] = 2.0f;
			    goal_vals[3] = 4.0f;
			    goal_vals[4] = 0.5f;
			    goal_vals[5] = 0.5f;
			    goal_vals[6] = 0.5f;
			    goal_vals[7] = 0.5f;
			    goal_vals[8] = 1.0f;
			    goal_vals[9] = 1.0f;
			    break;
	    	case 2:
			    goal_vals[0] = 1.0f;
			    goal_vals[1] = 1.0f;
			    goal_vals[2] = 1.0f;
			    goal_vals[3] = 2.0f;
			    goal_vals[4] = 2.0f;
			    goal_vals[5] = 4.0f;
			    goal_vals[6] = 0.5f;
			    goal_vals[7] = 0.5f;
			    goal_vals[8] = 0.5f;
			    goal_vals[9] = 0.5f;
			    break;
			default:
    	    	Log.info("ERROR: Unhandled context (" + Main.context + ")");
    			System.exit(0);
    	}
	    
	    float score = 0.0f;
	    
	    for (int i = 0; i < final_state.intentions.size(); i++)
	    {
	        score += goal_vals[i] * (final_state.isIntentionComplete(i) ? 1 : 0);
	    }
	    
	    if (normalise)
	    {
	    	score = (score - baseline_mean) / baseline_std;
	    }
	    
	    return score;
    }
}
