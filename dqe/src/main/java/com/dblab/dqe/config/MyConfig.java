package com.dblab.dqe.config;

import com.dblab.dqe.module.FullPipeline;
import net.sf.extjwnl.JWNLException;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;

@Configuration
public class MyConfig {

    @Bean
    public FullPipeline pipeline() throws IOException, JWNLException {
        return new FullPipeline();
    }
}
