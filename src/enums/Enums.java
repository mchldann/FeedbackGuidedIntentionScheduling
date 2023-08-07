package enums;

public class Enums {
	
	public static final int MAX_CONSECUTIVE_PASSES = 6;
	public static final boolean ALLOW_UNFORCED_PASS = false;
	
    public enum OracleType
    {
    	HUMAN_FEEDBACK
    };
    
    public enum RolloutEvaluationType
    {
    	DEFAULT,
        LEARNED
    };
    
    public enum ExperimentPhase
    {
    	BASELINE,
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
