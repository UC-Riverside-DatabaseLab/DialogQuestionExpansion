package com.dblab.dqe.models;

public class ExpansionOutput {
    private String originalQ;
    private String resolvedQ;

    public ExpansionOutput(String originalQ, String resolvedQ) {
        this.originalQ = originalQ;
        this.resolvedQ = resolvedQ;
    }

    public String getOriginalQ() {
        return originalQ;
    }

    public void setOriginalQ(String originalQ) {
        this.originalQ = originalQ;
    }

    public String getResolvedQ() {
        return resolvedQ;
    }

    public void setResolvedQ(String resolvedQ) {
        this.resolvedQ = resolvedQ;
    }
}
