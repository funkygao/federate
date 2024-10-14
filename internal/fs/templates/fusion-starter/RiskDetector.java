package {{.Package}};

public interface RiskDetector {
    void analyzeClass(Class<?> clazz);
    void reportRisks();
}

