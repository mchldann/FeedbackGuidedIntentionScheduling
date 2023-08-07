package enums;

public class Enums {
	
	public static final int MAX_CONSECUTIVE_PASSES = 6;
	public static final boolean ALLOW_UNFORCED_PASS = false;
	
    public enum OracleType
    {
        WEIGHTED_GOALS,
        WEIGHTED_GOALS_NOISY,
        
        CONTEXTUAL_WEIGHTED_GOALS,
        CONTEXTUAL_WEIGHTED_GOALS_NOISY,
        
        TIME_TAKEN,
        TIME_TAKEN_NOISY
    };
    
    public enum RolloutEvaluationType
    {
    	DEFAULT,
        ORACLE,
        LEARNED
    };
    
    public enum ExperimentPhase
    {
    	BASELINE,
        ORACLE,
        BURN_IN,
        LEARNING
    };
    
    public enum AllianceType
    {
        ADVERSARIAL,
        ALLIED,
        NEUTRAL,
        P1_HELPER,
        P1_MEDDLER,
        TWO_VS_TWO
    };
    
    public enum VisionType
    {
    	FULL,
    	UNAWARE,
    	PARTIALLY_AWARE
    }
}
