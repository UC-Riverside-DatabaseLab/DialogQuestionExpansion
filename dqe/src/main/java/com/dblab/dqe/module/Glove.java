package com.dblab.dqe.module;

import edu.stanford.nlp.neural.NeuralUtils;
import org.ejml.simple.SimpleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class Glove {

    private static final Logger logger = LoggerFactory.getLogger(Glove.class);

    private int dim;
    private Map<String, SimpleMatrix> vectorMap;
    private Set<String> stopWords;

    public Glove() throws IOException {
        logger.info("[DQE] Initializing glove...");
        this.dim = 50;
        this.vectorMap = new HashMap<>();
        this.stopWords = new HashSet<>();
        readFile();
    }

    public void readFile() throws IOException {
        File file = new File("src/main/resources/dataFile/glove.6B." + dim + "d.txt");
        File stopWordfile = new File("src/main/resources/dataFile/snowball.txt");
        Scanner sc = new Scanner(file);
        String line;
        while (sc.hasNextLine()) {
            line = sc.nextLine();
            String[] tokens = line.split("\\s");
            double[] vector = new double[dim];
            for (int i = 0; i < tokens.length - 1; i++) {
                vector[i] = Double.parseDouble(tokens[i + 1]);
            }
            SimpleMatrix s = new SimpleMatrix(1, dim, true, vector);
            vectorMap.put(tokens[0], s);
        }
        sc = new Scanner(stopWordfile);
        while (sc.hasNextLine()) {
            this.stopWords.add(sc.nextLine().trim());
        }
    }

    public double sim(String token1, String token2) {
        token1 = token1.toLowerCase();
        token2 = token2.toLowerCase();
        if (vectorMap.containsKey(token1) && vectorMap.containsKey(token2)) {
            SimpleMatrix s1 = vectorMap.get(token1);
            SimpleMatrix s2 = vectorMap.get(token2);
            return NeuralUtils.cosine(s1, s2);
        } else {
            return 0;
        }
    }

    public double sim2(String token1, String token2) {
        SimpleMatrix s1 = null, s2 = null;
        for (String t1 : token1.toLowerCase().split("[^a-zA-Z0-9']+")) {
            if (vectorMap.containsKey(t1) && !stopWords.contains(t1)) {
                if (s1 == null) {
                    s1 = vectorMap.get(t1);
                } else {
                    s1 = s1.plus(vectorMap.get(t1));
                }
            }
        }
        for (String t2 : token2.toLowerCase().split("[^a-zA-Z0-9']+")) {
            if (vectorMap.containsKey(t2) && !stopWords.contains(t2)) {
                if (s2 == null) {
                    s2 = vectorMap.get(t2);
                } else {
                    s2 = s2.plus(vectorMap.get(t2));
                }
            }
        }
        if (s1 != null && s2 != null) {
            return NeuralUtils.cosine(s1, s2);
        } else {
            return 0.0;
        }
    }

    public double sim2Max(String token1, String token2) {
        SimpleMatrix s1 = null, s2 = null;
        double max = Double.MIN_VALUE;
        for (String t1 : token1.toLowerCase().split("[^a-zA-Z0-9']+")) {
            if (vectorMap.containsKey(t1) && !stopWords.contains(t1)) {
                s1 = vectorMap.get(t1);
            }
            for (String t2 : token2.toLowerCase().split("[^a-zA-Z0-9']+")) {
                if (vectorMap.containsKey(t2) && !stopWords.contains(t2)) {
                    s2 = vectorMap.get(t2);
                }
                if (s1 != null && s2 != null) {
                    double sim = NeuralUtils.cosine(s1, s2);
                    if (sim > max) {
                        max = sim;
                    }
                }
            }
        }
        return max;
    }
}
