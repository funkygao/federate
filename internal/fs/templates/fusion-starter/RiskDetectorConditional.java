package {{.Package}};

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.annotation.AnnotationUtils;

import java.util.*;
import java.util.jar.JarFile;

class RiskDetectorConditional implements RiskDetector {
    private final Map<String, Set<String>> keys = new HashMap<>();

    @Override
    public void visit(JarFile jarFile, Class<?> clazz) throws Exception {
        ConditionalOnProperty conditionalOnProperty = AnnotationUtils.findAnnotation(clazz, ConditionalOnProperty.class);
        if (conditionalOnProperty == null) {
            return;
        }

        for (String key : conditionalOnProperty.value()) {
            if (!keys.containsKey(key)) {
                keys.put(key, new HashSet<>());
            }
            keys.get(key).add(clazz.getCanonicalName());
        }
    }

    @Override
    public String conflictDesc() {
        return "@ConditionalOnProperty";
    }

    @Override
    public void reportRisks() {
        for (Map.Entry<String, Set<String>> entry : keys.entrySet()) {
            System.out.printf("  %s\n", entry.getKey());
            for (String clazz : entry.getValue()) {
                System.out.printf("    %s\n", clazz);
            }
        }
    }
}

