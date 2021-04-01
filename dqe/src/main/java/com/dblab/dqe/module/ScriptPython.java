package com.dblab.dqe.module;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public class ScriptPython {

    private static final Logger logger = LoggerFactory.getLogger(ScriptPython.class);
    Process mProcess;

    public String runScript(String a) {
        Process process;
        try {
            process = Runtime.getRuntime().exec("python src/main/resources/dataFile/HuggingFace.py " + a);
            mProcess = process;
        } catch (Exception e) {
            logger.error("Exception Raised" + e.toString());
        }
        InputStream stdout = mProcess.getInputStream();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
        String line;
        try {
            while ((line = reader.readLine()) != null) {
//                System.out.println(line);
                return line;
            }
        } catch (IOException e) {
            logger.error("Exception in reading output" + e.toString());
        }
        return null;
    }
}

/*
 * Testing purpose only
 */
//class Solution {
//    public static void main(String[] args){
//      ScriptPython scriptPython = new ScriptPython();
//      System.out.println(scriptPython.runScript("\"\"\"Anything interesting happening in Toronto, Canada? What do you have in mind? Music? Sports? Check out some Sports. Well, there's the Blue Jays Vs Astros at 6:30 pm, March 13th at the Rogers Centre. Are there any All-Stars on either of those teams?\"\"|-4$-2|-12$-11|-6$-5|-11$-7|-11$-2|-6$-2"));
//    }
//}

