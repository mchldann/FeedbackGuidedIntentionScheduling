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
	public boolean uses_coins;
	public boolean uses_enemies;
	public int max_trajectory_length;

    public Oracle(OracleType type)
    {
        this.type = type;
        
    	switch (type)
    	{
    		case HUMAN_FEEDBACK:
    			noisy = false;
    			uses_time = true;
    			uses_coins = true;
    			uses_enemies = true;
    			max_trajectory_length = 48;
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
    		case HUMAN_FEEDBACK:
    			return 0.0f;
    		default:
    	    	Log.info("ERROR: Unhandled oracle type (" + type.toString() + ")");
    			System.exit(0);
    			return Float.NEGATIVE_INFINITY;
    	}
    }
}
