package com.dblab.dqe.controllers;

import com.dblab.dqe.models.ExpansionOutput;
import com.dblab.dqe.module.FullPipeline;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.sf.extjwnl.JWNLException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
@CrossOrigin("*")
public class ExpansionController {

    private static final Logger logger = LoggerFactory.getLogger(ExpansionController.class);

    private final FullPipeline pipeline;

    @Autowired
    public ExpansionController(FullPipeline pipeline) {
        this.pipeline = pipeline;
    }

    @PostMapping("/expand")
    public ExpansionOutput getExpansion(@RequestBody String dialog) throws JWNLException, CloneNotSupportedException, IOException {
        JsonObject dialogObj = JsonParser.parseString(dialog).getAsJsonObject();
        return pipeline.resolve(dialogObj);
    }
}
