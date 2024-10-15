package {{.Package}};

import java.util.jar.JarFile;

public interface RiskDetector {
    void visit(JarFile jarFile, Class<?> clazz) throws Exception;

    void reportRisks();

    String conflictDesc();
}

