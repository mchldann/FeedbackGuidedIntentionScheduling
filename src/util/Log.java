package util;

import java.io.File;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Handler;
import java.util.logging.Logger;

public class Log {
	
	public static boolean log_to_file = true;
	
    private static String logDirPath;
    private static Logger logger;
    
    public static void refreshLogDir(String base_dir, boolean add_subdir)
    {
    	if (add_subdir)
    	{
	    	Date date = Calendar.getInstance().getTime();
	        DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
	        logDirPath = base_dir + "/" + dateFormat.format(date);
    	}
    	else
    	{
    		logDirPath = base_dir;
    	}
    	
    	new File(logDirPath).mkdirs();
    }
    
    public static void info(String message)
    {
    	info(message, true);
    }
    
    public static void info(String message, boolean verbose)
    {
    	if (!verbose)
    	{
    		return;
    	}
    	
    	if (logger == null)
    	{
    		logger = Logger.getLogger("util.Log");
    		logger.setUseParentHandlers(false);
    		
			try
			{
				if (log_to_file)
				{
					Handler handler = new FileHandler(logDirPath + "/log.txt");
		            handler.setFormatter(new CustomRecordFormatter());
		            logger.addHandler(handler);
				}
	            
	            Handler consoleHandler = new ConsoleHandler();
	            consoleHandler.setFormatter(new CustomRecordFormatter());
	            logger.addHandler(consoleHandler);
			}
			catch (SecurityException | IOException e)
			{
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}

        logger.info(message + "\n");
    }
    
    public static String getLogDir()
    {
    	return logDirPath;
    }
}
