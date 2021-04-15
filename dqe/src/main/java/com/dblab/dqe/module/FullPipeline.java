package com.dblab.dqe.module;

import com.dblab.dqe.models.ExpansionOutput;
import com.google.api.client.http.*;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Constituent;
import edu.stanford.nlp.trees.LabeledScoredConstituentFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import net.sf.extjwnl.JWNLException;
import net.sf.extjwnl.data.IndexWord;
import net.sf.extjwnl.data.POS;
import net.sf.extjwnl.data.PointerType;
import net.sf.extjwnl.data.relationship.RelationshipFinder;
import net.sf.extjwnl.data.relationship.RelationshipList;
import net.sf.extjwnl.dictionary.Dictionary;
import org.apache.commons.lang3.StringUtils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

public class FullPipeline {

    private static final Logger logger = LoggerFactory.getLogger(FullPipeline.class);

    private Glove glove;
    private Set<String> stopWords;
    private List<String> wordFreqList;
    private StanfordCoreNLP stanfordCoreNLP;
    private Dictionary dictionary;

    private JsonObject dialogObj;
    private String question;
    private String wholeDialog;
    private Map<String, String> keyValuePairBot;
    private Map<String, String> keyValuePairUser;
    private List<String> NPs; // Noun Phrases
    private Map<String, LinkedHashMap<String, Double>> corefScore;
    private String corefQuery;
    private String corefStateReplaced;
    private String ellipsisQuery;
    private Map<Double, String> ellipsisRanking;

    public FullPipeline() throws IOException, JWNLException {
        init();
    }

    private void init() throws IOException, JWNLException {
        logger.info("Initializing pipeline...");

        glove = new Glove();
        readStopWords();
        readWordFreq();
        initStanfordNLP();
        initWordNetDict();

        logger.info("Pipeline initialization finished...");
    }

    private void readStopWords() throws FileNotFoundException {
        File f = new File("src/main/resources/dataFile/snowball.txt");
        stopWords = new HashSet<>();
        Scanner sc = new Scanner(f);
        while (sc.hasNextLine()) {
            stopWords.add(sc.nextLine());
        }
    }

    private void readWordFreq() throws FileNotFoundException {
        File f = new File("src/main/resources/dataFile/unigram_freq.csv");
        Scanner sc = new Scanner(f);
        String line = sc.nextLine();
        wordFreqList = new ArrayList<>();
        while (sc.hasNextLine()) {
            line = sc.nextLine();
            wordFreqList.add(line.substring(0, line.indexOf(",")));
        }
    }

    private void initStanfordNLP() {
        Properties nlpProps = new Properties();
        nlpProps.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref,kbp");
        stanfordCoreNLP = new StanfordCoreNLP(nlpProps);
    }

    private void initWordNetDict() throws JWNLException {
        dictionary = Dictionary.getDefaultResourceInstance();
    }

    public ExpansionOutput resolve(JsonObject dialogObj) throws JWNLException, CloneNotSupportedException, IOException {
        reset(dialogObj);
        logger.info("Starting coref...");
        coref();
        compareState();
        corefOutput();
        if (!corefQuery.equalsIgnoreCase(question)) {
            return new ExpansionOutput(question, corefQuery);
        }
        logger.info("Starting ellipsis...");
        ellipsis();

        return new ExpansionOutput(question, ellipsisQuery);
    }

    private void reset(JsonObject obj) {
        keyValuePairBot = new HashMap<>();
        keyValuePairUser = new HashMap<>();
        NPs = new ArrayList<>();
        corefScore = new HashMap<>();
        dialogObj = obj;
        question = dialogObj.get("question").getAsString();
        corefQuery = question;
        ellipsisQuery = question;
        corefStateReplaced = "";
        ellipsisRanking = new TreeMap<>();
        JsonArray turns = dialogObj.get("turns").getAsJsonArray();
        String text = "";
        for (int i = 0; i < turns.size() - 1; i++) {
            String utterance = turns.get(i).getAsJsonObject().get("utterance").getAsString();
            text += utterance + " ";
            JsonArray frames = turns.get(i).getAsJsonObject().getAsJsonArray("frames");
            for (int j = 0; j < frames.size(); j++) {
                JsonObject frame = frames.get(j).getAsJsonObject();
                if (frame.has("actions")) {
                    JsonArray actions = frame.getAsJsonArray("actions");
                    for (int k = 0; k < actions.size(); k++) {
                        JsonObject action = actions.get(k).getAsJsonObject();
                        String slot = action.get("slot").getAsString();
                        JsonArray values = action.getAsJsonArray("values");
                        if (values.size() == 1) {
                            String value = values.get(0).getAsString();
                            keyValuePairBot.put(slot, value);
                        }
                    }
                }
                if (frame.has("state")) {
                    JsonObject slotValues = frame.getAsJsonObject("state").getAsJsonObject("slot_values");
                    for (String key : slotValues.keySet()) {
                        String value = slotValues.get(key).getAsJsonArray().get(0).getAsString();
                        keyValuePairUser.put(key, value);
                    }
                }
            }
        }
        text += question;
        wholeDialog = text;
    }

    private void coref() {
        CoreDocument document = new CoreDocument(question);
        stanfordCoreNLP.annotate(document);

        CoreSentence lastSentence = document.sentences().get(document.sentences().size() - 1);
        Tree constituencyParse = lastSentence.constituencyParse();
        Set<Constituent> treeConstituents = constituencyParse.constituents(new LabeledScoredConstituentFactory());

        String query = "\"\"\"" + wholeDialog + "\"\"|";
        NPs = new ArrayList<>();
        for (Constituent constituent : treeConstituents) {
            if (constituent.label() != null) {
                if (constituent.label().toString().equals("NP")) {
                    String nounPhrase = "";
                    for (int i = constituent.start(); i < constituent.end() + 1; i++) {
                        nounPhrase += constituencyParse.getLeaves().get(i).value() + " ";
                    }
                    nounPhrase = nounPhrase.trim();
                    NPs.add(nounPhrase);
                    int start = constituent.start() - lastSentence.tokens().size() - 1;
                    int end = constituent.end() - lastSentence.tokens().size();
                    query += start + "$" + end + "|";
                }
            }
        }

        List<CoreLabel> pronouns = constituencyParse.taggedLabeledYield();
        for (CoreLabel label : pronouns) {
            if (label.value().contains("PRP")) {
                int startPos = label.get(CoreAnnotations.IndexAnnotation.class);
                int start = startPos - lastSentence.tokens().size() - 1;
                int end = start + 1;
                if (!NPs.contains(label.word())) {
                    NPs.add(label.word());
                    query += start + "$" + end + "|";
                }
            }
        }

        Set<String> toBeRemoved = new HashSet<>();
        if (!NPs.isEmpty()) {
            ScriptPython scriptPython = new ScriptPython();
            query = query.substring(0, query.length() - 1);
            String corefScoreString = scriptPython.runScript(query);
            int i = 0;
            if (corefScoreString != null) {
                for (String score : corefScoreString.split("\\|")) {
                    if (!score.equals("None")) {
                        corefScore.put(NPs.get(i), parseHuggingFaceScore(corefScoreString));
                    } else {
                        toBeRemoved.add(NPs.get(i));
                    }
                    i++;
                }
            }
        }
        for (String toBeRemove : toBeRemoved) {
            if (NPs.contains(toBeRemove)) {
                NPs.remove(toBeRemove);
            }
        }
    }

    private void compareState() throws JWNLException, CloneNotSupportedException, IOException {
//        FileWriter ltrw = new FileWriter(LEARN_TO_RANK_FILE,true);
//        BufferedWriter ltrWriter = new BufferedWriter(ltrw);
//        ltrWriter.write("# Query " + (i+1) + " # truth: " + truth);
//        ltrWriter.newLine();
        List<String> nounPhrases = getNP(question);
        List<String> toBeRemoved = new ArrayList<>();
        List<String> PRPs = Arrays.asList("he", "him", "his", "her", "she");
        List<String> nonPersons = Arrays.asList("it", "its", "they", "them", "their", "that", "those");
//        ScriptPythonForGPT2 scriptPython = new ScriptPythonForGPT2();
        int count2 = 0;
        for (String nounPhrase : corefScore.keySet()) {
            if (nounPhrases.contains(nounPhrase)) {
                LinkedHashMap<String, Double> map = corefScore.get(nounPhrase);
                LinkedHashMap<String, Double> unsorted = new LinkedHashMap<>();
                if (PRPs.contains(nounPhrase.toLowerCase())) { // hugging face is able to handle personal pronouns
                    for (String word : map.keySet()) {
                        String stateValue = word.split("\\|")[1];
                        String key = word.split("\\|")[0];
//                        key = stateConversion(key);
                        double kb = 0.0;
                        if (key.equalsIgnoreCase("event_name")) {
                            kb = KB(stateValue, "person");
                        }
                        unsorted.put(word, kb);
                        String GPT2String;
                        if (kb > 0.0) {
                            GPT2String = question.replace(nounPhrase, "person");
                        } else {
                            GPT2String = question.replace(nounPhrase, key);
                        }
//                        String GPTScore = scriptPython.runScript("\"" + GPT2String + "\"");
                        String newQ = question.replace(nounPhrase, stateValue);
//                        if (truth.toLowerCase().contains(stateValue.toLowerCase())) {
//                            if (caseType.equalsIgnoreCase("coref")) {
//                                ltrWriter.write("2 qid:" + (i + 1) + " 1:0 2:0 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                if (count2 > 1) {
//                                    ltrWriter.write(" #####");
//                                }
//                                count2++;
//                            } else {
//                                ltrWriter.write("1 qid:" + (i + 1) + " 1:0 2:0 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                            }
//                        } else {
//                            ltrWriter.write("0 qid:" + (i + 1) + " 1:0 2:0 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                        }
//                        ltrWriter.newLine();
                    }
                } else if (nonPersons.contains(nounPhrase.toLowerCase())) {
                    CoreDocument document = new CoreDocument(question);
                    stanfordCoreNLP.annotate(document);
                    CoreSentence lastSentence = document.sentences().get(0);
                    List<String> nounPhrasesSent = lastSentence.nounPhrases();
                    List<String> verbPhrases = lastSentence.verbPhrases();
                    String keyword = getKeyword(question).getLemma();
                    if (keyword != null) {
//                        debugWriter.write("Keyword: " + keyword);
//                        debugWriter.newLine();
                        for (String word : map.keySet()) {
                            String key = word.split("\\|")[0];
//                            key = stateConversion(key);
                            double depth = 5.0 / depth(key, keyword);
                            double sim = glove.sim2Max(key, keyword);
                            double score = (depth + sim) / 2;
                            double kb = 0.0;
                            String stateValue = word.split("\\|")[1];
                            if (key.equalsIgnoreCase("event")) {
                                kb = KB(stateValue, keyword);
                            }
                            unsorted.put(word, score);
                            String GPT2String;
                            if (nounPhrase.toLowerCase().trim().startsWith("the")) {
                                GPT2String = question.replace(nounPhrase, "the " + key);
                            } else {
                                GPT2String = question.replace(nounPhrase, key);
                            }
//                            String GPTScore = scriptPython.runScript("\"" + GPT2String + "\"");
                            String newQ = question.replace(nounPhrase, stateValue);
//                            if (truth.toLowerCase().contains(stateValue.toLowerCase())) {
//                                if (caseType.equalsIgnoreCase("coref")) {
//                                    ltrWriter.write("2 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                    if (count2 > 1) {
//                                        ltrWriter.write(" #####");
//                                    }
//                                    count2++;
//                                } else {
//                                    ltrWriter.write("1 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                }
//                            } else {
//                                ltrWriter.write("0 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) +" 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                            }
//                            ltrWriter.newLine();
                        }
                    }
                } else {
                    for (String word : map.keySet()) {
                        String key = word.split("\\|")[0];
                        String stateValue = word.split("\\|")[1];;
//                        key = stateConversion(key);
                        double depth = 5.0 / depth(key, nounPhrase);
                        double sim = glove.sim2Max(key, nounPhrase);
                        double kb;
                        double score;
                        String GPT2String;
                        if (nounPhrase.toLowerCase().trim().startsWith("the")) {
                            GPT2String = question.replace(nounPhrase, "the " + key);
                        } else {
                            GPT2String = question.replace(nounPhrase, key);
                        }
//                        String GPTScore = scriptPython.runScript("\"" + GPT2String + "\"");
                        String newQ = question.replace(nounPhrase, stateValue);
                        if (key.equalsIgnoreCase("event_name")) {
                            kb = KB(key, nounPhrase);
                            score = (depth + sim + kb) / 3;
//                            if (truth.toLowerCase().contains(stateValue.toLowerCase())) {
//                                if (caseType.equalsIgnoreCase("coref")) {
//                                    ltrWriter.write("2 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                    if (count2 > 1) {
//                                        ltrWriter.write(" #####");
//                                    }
//                                    count2++;
//                                } else {
//                                    ltrWriter.write("1 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                }
//                            } else {
//                                ltrWriter.write("0 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) +" 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                            }
//                            ltrWriter.newLine();
                        } else {
                            score = (depth + sim) / 2;
//                            if (truth.toLowerCase().contains(stateValue.toLowerCase())) {
//                                if (caseType.equalsIgnoreCase("coref")) {
//                                    ltrWriter.write("2 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) + " 2:" + String.format("%.4f", sim) + " 3:0" + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                    if (count2 > 1) {
//                                        ltrWriter.write(" #####");
//                                    }
//                                    count2++;
//                                } else {
//                                    ltrWriter.write("2 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) + " 2:" + String.format("%.4f", sim) + " 3:0" + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                                }
//                            } else {
//                                ltrWriter.write("0 qid:" + (i + 1) + " 1:" + String.format("%.4f", depth / 5) +" 2:" + String.format("%.4f", sim) + " 3:0" + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                            }
//                            ltrWriter.newLine();
                        }
                        unsorted.put(word, score);
                    }
                }
                LinkedHashMap<String, Double> sortedMap = new LinkedHashMap<>();
                unsorted.entrySet()
                        .stream()
                        .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                        .forEachOrdered(x -> sortedMap.put(x.getKey(), x.getValue()));

                corefScore.put(nounPhrase, sortedMap);
            } else {
                toBeRemoved.add(nounPhrase);
            }
        }
        for (String s : toBeRemoved) {
            corefScore.remove(s);
        }
//        ltrWriter.close();
//        ltrw.close();
    }

    private void corefOutput() throws IOException { // prioritize pronoun and then output the highest score
        String newQ = question;
        List<String> pronouns = Arrays.asList("he", "him", "his", "her", "she", "it", "its", "they", "them", "their", "that", "those");
        if (!corefScore.isEmpty()) {
            // first run pronouns
            boolean needSecondRun = true;
            for (String replacement : corefScore.keySet()) {
                if (pronouns.contains(replacement.toLowerCase().trim())) {
                    LinkedHashMap<String, Double> scores = corefScore.get(replacement);
                    for (String s : scores.keySet()) {
                        String[] tokenss = s.split("\\|");
                        corefStateReplaced = tokenss[0];
                        if (replacement.toLowerCase().trim().startsWith("the ")) {
                            this.corefStateReplaced = "the " + this.corefStateReplaced;
                        }
//                        this.corefQueryForGPT2 = newQ.replace(replacement, this.corefStateReplaced);
                        newQ = newQ.replace(replacement, tokenss[1]);
                        break;
                    }
                    needSecondRun = false;
                }
            }
            // second run highest score;
            if (needSecondRun) {
                double highScore = 0.0;
                String replacementString = "";
                String replacementToBeString = "";
                for (String replacement : corefScore.keySet()) {
                    LinkedHashMap<String, Double> scores = corefScore.get(replacement);
                    for (String s : scores.keySet()) {
                        if (scores.get(s) > highScore) {
                            String[] tokens = s.split("\\|");
                            corefStateReplaced = tokens[0];
                            highScore = scores.get(s);
                            replacementString = replacement;
                            replacementToBeString = s.split("\\|")[1];
                        }
                        break;
                    }
                }
                if (replacementString.toLowerCase().trim().startsWith("the ")) {
                    corefStateReplaced = "the " + corefStateReplaced;
                }
                if (!replacementString.isEmpty()) {
//                    this.corefQueryForGPT2 = newQ.replace(replacementString, this.corefStateReplaced);
                    newQ = newQ.replace(replacementString, replacementToBeString);
                }
            }
        }
        corefQuery = newQ;
//        this.GPT2String += "\"" + this.corefQueryForGPT2;
    }

    private void ellipsis() throws IOException, JWNLException, CloneNotSupportedException {
//        FileWriter ltrw = new FileWriter(LEARN_TO_RANK_FILE,true);
//        BufferedWriter ltrWriter = new BufferedWriter(ltrw);
        Map<String, String> keyValuePair = new HashMap<>();
        if (!question.endsWith("?")) {
            if (question.endsWith(".")) {
                question = question.substring(0, question.length() - 1);
            }
            question += "?";
        }
//        String truth = dialogObj.get("truth").getAsString();
        JsonArray turns = dialogObj.get("turns").getAsJsonArray();
        for (int i = 0; i < turns.size() - 1; i++) {
            JsonArray frames = turns.get(i).getAsJsonObject().getAsJsonArray("frames");
            for (int j = 0; j < frames.size(); j++) {
                JsonObject frame = frames.get(j).getAsJsonObject();
                if (frame.has("actions")) {
                    JsonArray actions = frame.getAsJsonArray("actions");
                    for (int k = 0; k < actions.size(); k++) {
                        JsonObject action = actions.get(k).getAsJsonObject();
                        String slot = action.get("slot").getAsString();
                        if (!slot.equalsIgnoreCase("count")) {
                            JsonArray values = action.getAsJsonArray("values");
                            if (values.size() == 1) {
                                String value = values.get(0).getAsString();
                                keyValuePair.put(slot, value);
                            }
                        }
                    }
                }
                if (frame.has("state")) {
                    JsonObject slotValues = frame.getAsJsonObject("state").getAsJsonObject("slot_values");
                    for (String key : slotValues.keySet()) {
                        String value = slotValues.get(key).getAsJsonArray().get(0).getAsString();
                        keyValuePair.put(key, value);
                    }
                }
            }
        }

        IndexWord keyword = getKeyword(question);
        String ellipsis = "";
//        ScriptPythonForGPT2 scriptPython = new ScriptPythonForGPT2();
        if (keyword.getPOS().equals(POS.VERB)) {
            double maxScore = 0.0;
            for (String key : keyValuePair.keySet()) {
                double sim = glove.sim2Max(key, keyword.getLemma());
//                System.out.println(keyword + " " + key + ": " + dist);
                double kb = 0.0;
                if (key.equalsIgnoreCase("event_name")) {
                    kb = KB(keyValuePair.get(key), keyword.getLemma());
                }
                double score = (sim + kb) / 2;
                if (score > maxScore) {
                    maxScore = score;
                    ellipsis = keyValuePair.get(key);
                }
//                String GPT2String = question.substring(0, question.length() - 1) + " " + key + "?";
//                String GPTScore = scriptPython.runScript("\"" + GPT2String + "\"");
//                String newQ = question.substring(0, question.lastIndexOf("?")) + " " + keyValuePair.get(key) + "?";
//                if (truth.toLowerCase().contains(keyValuePair.get(key).toLowerCase())) {
//                    if (caseType.equalsIgnoreCase("ellipsis")) {
//                        ltrWriter.write("2 qid:" + (kk + 1) + " 1:0 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                    } else {
//                        ltrWriter.write("1 qid:" + (kk + 1) + " 1:0 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                    }
//                } else {
//                    ltrWriter.write("0 qid:" + (kk + 1) + " 1:0 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                }
//                ltrWriter.newLine();
            }
        } else if (keyword.getPOS().equals(POS.NOUN)) {
            double maxScore = 0.0;
            String prevKey = "";
            for (String key : keyValuePair.keySet()) {
                IndexWord keyWord = dictionary.lookupIndexWord(POS.NOUN, key);
                if (key.equalsIgnoreCase("category") || key.equalsIgnoreCase("subcategory")) {
                    keyWord = dictionary.lookupIndexWord(POS.NOUN, keyValuePair.get(key));
                }
                int dist = 1000;
                try {
                    RelationshipList list = RelationshipFinder.findRelationships(keyWord.getSenses().get(0), keyword.getSenses().get(0), PointerType.HYPERNYM);
                    dist = list.get(0).getDepth();
                } catch (NullPointerException e) {

                }
                double sim = glove.sim2Max(key, keyword.getLemma());
                double score = (5.0 / dist + sim) / 2;
                if (key.equalsIgnoreCase("category") || key.equalsIgnoreCase("subcategory")) {
                    sim = glove.sim2Max(keyValuePair.get(key), keyword.getLemma());
                    score = (5.0 / dist + sim) / 2;
                }
                if (key.equalsIgnoreCase("event_name")) {
                    score = (5.0 / dist + glove.sim2Max(key, keyword.getLemma()) + KB(keyValuePair.get(key), keyword.getLemma())) / 3;
                }
//                double kb = 0.0;
//                if (key.equalsIgnoreCase("event_name")) {
//                    kb = KB(keyValuePair.get(key), keyWord.getLemma());
//                }
//                String GPT2String = question.substring(0, question.length() - 1) + " " + key + "?";
//                String GPTScore = scriptPython.runScript("\"" + GPT2String + "\"");
//                String newQ = question.substring(0, question.lastIndexOf("?")) + " " + keyValuePair.get(key) + "?";
//                if (truth.toLowerCase().contains(keyValuePair.get(key).toLowerCase())) {
//                    if (caseType.equalsIgnoreCase("ellipsis")) {
//                        ltrWriter.write("2 qid:" + (kk + 1) + " 1:" + String.format("%.4f", 1.0 / dist) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                    } else {
//                        ltrWriter.write("1 qid:" + (kk + 1) + " 1:" + String.format("%.4f", 1.0 / dist) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                    }
//                } else {
//                    ltrWriter.write("0 qid:" + (kk + 1) + " 1:"+ String.format("%.4f", 1.0 / dist) + " 2:" + String.format("%.4f", sim) + " 3:" + String.format("%.4f", kb) + " 4:" + String.format("%.4f", 1.0 / Double.parseDouble(GPTScore.substring(1, 7))) + " # " + newQ);
//                }
//                ltrWriter.newLine();
                if (score > maxScore) {
                    maxScore = score;
                    ellipsis = keyValuePair.get(key);
//                    ellipsisStateReplaced = key;
                }
            }
        } else {
            throw new IllegalArgumentException("Bad question: " + question);
        }
//        System.out.println(question + " ----------------- " + question.substring(0, question.lastIndexOf("?")).replace(keyword.getLemma(), keyword.getLemma().toUpperCase()) + " " + ellipsis + "? ----------------- " + truth);
//        BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE, true));
//        if (!this.corefReplace) {
//            writeFile.write(question.substring(0, question.lastIndexOf("?")) + " " + ellipsis.toLowerCase() + "?|" + "ellipsis");
//        }
        String newQ = question.substring(0, question.lastIndexOf("?")) + " " + ellipsis + "?";
        ellipsisQuery = newQ;
//        this.ellipsisQueryForGPT2 = question.substring(0, question.lastIndexOf("?")) + " " + this.ellipsisStateReplaced + "?";
//        writer.write(newQ);
//        this.GPT2String += "|" + this.ellipsisQueryForGPT2 + "|" + this.truth + "\"";
//        this.GPT2String = this.GPT2String.toLowerCase();
//        ScriptPythonForGPT2 scriptPython = new ScriptPythonForGPT2();
//        System.out.println(this.GPT2String); //coref|ellipsis|truth
//        String corefScoreString = scriptPython.runScript(this.GPT2String);
    }

    private LinkedHashMap<String, Double> parseHuggingFaceScore(String s) {
        Map<String, Double> unsorted = new HashMap<>();
        s = s.substring(s.indexOf(", ") + 1 , s.lastIndexOf("}")).trim();
        String[] tokens = s.split(", ");
        String prevToken = "";
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].contains(":")) {
                String[] scores = tokens[i].split(":");
                String score = scores[scores.length - 1];
                try {
                    double scoreDouble = Double.parseDouble(score);
                    if (!prevToken.isEmpty()) {
                        String key = prevToken;
                        for (int j = 0; j < scores.length - 1; j++) {
                            key += scores[j] + ":";
                        }
                        String keyString = foundState(key.substring(0, key.lastIndexOf(":")));
                        if (!keyString.isEmpty()) {
                            unsorted.put(keyString + "|" + key.substring(0, key.lastIndexOf(":")), scoreDouble);
                        }
                        prevToken = "";
                    } else {
                        String key = "";
                        for (int j = 0; j < scores.length - 1; j++) {
                            key += scores[j] + ":";
                        }
                        String keyString = foundState(key.substring(0, key.lastIndexOf(":")));
                        if (!keyString.isEmpty()) {
                            unsorted.put(keyString + "|" + key.substring(0, key.lastIndexOf(":")), scoreDouble);
                        }
                    }
                } catch (NumberFormatException e) {
                    prevToken += tokens[i] + ", ";
                }
            } else {
                prevToken += tokens[i] + ", ";
            }
        }
        LinkedHashMap<String, Double> sortedMap = new LinkedHashMap<>();
        unsorted.entrySet()
        .stream()
        .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
        .forEachOrdered(x -> sortedMap.put(x.getKey(), x.getValue()));

        return sortedMap;
//        double[] scoreArray = new double[sortedMap.size()];
//        int i = 0;
//        for (String noun : sortedMap.keySet()) {
//            scoreArray[i] = sortedMap.get(noun);
//            i++;
//        }
//        double[] metrics = calculateSD(scoreArray);
//
//        LinkedHashMap<String, Double> sortedNormalizedMap = new LinkedHashMap<>();
//        for (String ss : sortedMap.keySet()) {
//            if (sortedMap.size() == 1) {
//                sortedNormalizedMap.put(ss, 1.0);
//            } else {
//                sortedNormalizedMap.put(ss, (sortedMap.get(ss) - metrics[0]) / metrics[1]); // Z-score
//            }
//        }
//
//        return sortedNormalizedMap;
    }

    private String foundState(String word) {
        String found = null;
        String stateFound = "";
        for (String state : keyValuePairBot.keySet()) {
            String stateValue = keyValuePairBot.get(state);
            if (stateValue.equalsIgnoreCase(word) || stateValue.toLowerCase().replace("the", "").trim()
                    .equalsIgnoreCase(word.toLowerCase().replace("the", "").trim())) {
                found = "bot";
                stateFound = state;
            }
        }
        if (found == null) {
            for (String state : keyValuePairUser.keySet()) {
                String stateValue = keyValuePairUser.get(state);
                if (stateValue.equalsIgnoreCase(word) || stateValue.toLowerCase().replace("the", "").trim()
                        .equalsIgnoreCase(word.toLowerCase().replace("the", "").trim())) {
                    found = "user";
                    stateFound = state;
                }
            }
        }
        return stateFound;
    }

    private List<String> getNP(String s) {

        if (NPs.size() == 1) return NPs;

        boolean notDone = true;
        while (notDone) {
            notDone = false;
            for (int i = 0; i < NPs.size() - 1; i++) {
                boolean removed = false;
                for (int j = i + 1; j < NPs.size(); j++) {
                    if (NPs.get(i).contains(NPs.get(j))) {
                        corefScore.remove(NPs.get(i));
                        NPs.remove(i);
                        removed = true;
                        break;
                    } else if (NPs.get(j).contains(NPs.get(i))) {
                        corefScore.remove(NPs.get(j));
                        NPs.remove(j);
                        removed = true;
                        break;
                    }
                }
                if (removed) {
                    notDone = true;
                    break;
                }
            }
        }
        return NPs;
    }

    private double KB(String query, String nounPhrase) {
        CoreDocument document = new CoreDocument(query);
        stanfordCoreNLP.annotate(document);
        CoreSentence lastSentence = document.sentences().get(0);

        List<String> NPs = lastSentence.nounPhrases();
        double maxSim = 0.0;

        for (String np : NPs) {
            np = np.replaceAll("\\s", "+");
            try {
                HttpTransport httpTransport = new NetHttpTransport();
                HttpRequestFactory requestFactory = httpTransport.createRequestFactory();
                org.json.simple.parser.JSONParser parser = new JSONParser();
                GenericUrl url = new GenericUrl("https://kgsearch.googleapis.com/v1/entities:search");
                url.put("query", np);
                url.put("limit", "5");
                url.put("indent", "true");
                url.put("key", "AIzaSyAyEwSR9CuRye2UvnADxEKPE3_304eKwio");
                HttpRequest request = requestFactory.buildGetRequest(url);
                HttpResponse httpResponse = request.execute();
                JSONObject response = (JSONObject) parser.parse(httpResponse.parseAsString());
                JSONArray elements = (JSONArray) response.get("itemListElement");
                for (Object element : elements) {
                    JSONObject obj = (JSONObject) element;
                    JSONObject result = (JSONObject) obj.get("result");
                    JSONArray types = (JSONArray) result.get("@type");
                    for (Object s : types) {
                        String type = (String) s;
                        if (!type.equalsIgnoreCase("thing")) {
                            String[] typeKeys = StringUtils.splitByCharacterTypeCamelCase(type);
                            for (String k : typeKeys) {
                                if (!k.equalsIgnoreCase("or")) {
                                    double sim = glove.sim2Max(k, nounPhrase);
                                    if (sim > maxSim) {
                                        maxSim = sim;
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return maxSim;
    }

    private int depth(String a, String b) throws JWNLException, CloneNotSupportedException {
        int minDepth = 1000;
        for (String t1 : a.toLowerCase().split("[^a-zA-Z0-9']+")) {
            for (String t2 : b.toLowerCase().split("[^a-zA-Z0-9']+")) {
                IndexWord start = dictionary.lookupIndexWord(POS.NOUN, t1);
                IndexWord end = dictionary.lookupIndexWord(POS.NOUN, t2);
                if (start != null && end != null) {
                    RelationshipList list = RelationshipFinder.findRelationships(start.getSenses().get(0), end.getSenses().get(0), PointerType.HYPERNYM);
                    int depth = list.get(0).getDepth() + 1;
                    if (depth < minDepth) {
                        minDepth = depth;
                    }
                }
            }
        }

        return minDepth;
    }

    private IndexWord getKeyword(String question) throws JWNLException {
        Annotation annotation = new Annotation(question);
        stanfordCoreNLP.annotate(annotation);
        List<CoreMap> coreMap = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        Tree tree = coreMap.get(coreMap.size() - 1).get(TreeCoreAnnotations.TreeAnnotation.class);
        List<CoreLabel> labels = tree.taggedLabeledYield();
        IndexWord indexedWord = null;
        IndexWord badWord = null;
        int maxRank = 0;
        int badMaxRank = 0;
        for (int i = 0; i < labels.size(); i++) {
            CoreLabel label = labels.get(i);
            if (label.value().contains("NN") || label.value().contains("VB") ) {
                int rank = wordFreqList.indexOf(label.word());
                if (rank > maxRank) {
                    IndexWord keywordIndexed = dictionary.lookupIndexWord(POS.NOUN, label.word());
                    if (keywordIndexed != null) {
                        indexedWord = keywordIndexed;
                        maxRank = rank;
                    } else {
                        keywordIndexed = dictionary.lookupIndexWord(POS.VERB, label.word());
                        if (keywordIndexed != null) {
                            indexedWord = keywordIndexed;
                            maxRank = rank;
                        }
                    }
                }
            } else if (!label.value().equals(".")) {
                int rank = wordFreqList.indexOf(label.word());
                if (rank > badMaxRank) {
                    IndexWord keywordIndexed = dictionary.lookupIndexWord(POS.NOUN, label.word());
                    if (keywordIndexed != null) {
                        badWord = keywordIndexed;
                        badMaxRank = rank;
                    } else {
                        keywordIndexed = dictionary.lookupIndexWord(POS.VERB, label.word());
                        if (keywordIndexed != null) {
                            badWord = keywordIndexed;
                            badMaxRank = rank;
                        }
                    }
                }
            }
        }
        if (indexedWord != null) {
            return indexedWord;
        }
        return badWord;
    }

}
