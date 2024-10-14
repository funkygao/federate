package {{.Package}};

import lombok.extern.slf4j.Slf4j;
import org.springframework.core.annotation.AnnotationUtils;
import org.springframework.web.bind.annotation.RequestMapping;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

@Slf4j
class RequestMappingConflictDetector implements RiskDetector {
    private final Map<String, Set<String>> mappings = new HashMap<>();

    @Override
    public void analyzeClass(Class<?> clazz) {
        try {
            RequestMapping classMapping = AnnotationUtils.findAnnotation(clazz, RequestMapping.class);
            String[] classPaths = classMapping != null ? classMapping.value() : new String[]{""};

            for (Method method : clazz.getDeclaredMethods()) {
                try {
                    RequestMapping methodMapping = AnnotationUtils.findAnnotation(method, RequestMapping.class);
                    if (methodMapping != null) {
                        String[] methodPaths = methodMapping.value();
                        for (String classPath : classPaths) {
                            for (String methodPath : methodPaths) {
                                String fullPath = combinePaths(classPath, methodPath);
                                addMapping(fullPath, clazz.getName() + "#" + method.getName());
                            }
                        }
                    }
                } catch (Throwable t) {
                    log.trace("Error processing method {} in class {}: {}", method.getName(), clazz.getName(), t.getMessage());
                }
            }

            if (classMapping != null && classPaths.length > 0) {
                for (String classPath : classPaths) {
                    addMapping(classPath, clazz.getName());
                }
            }
        } catch (Throwable t) {
            log.trace("Error processing class {}: {}", clazz.getName(), t.getMessage());
        }
    }

    private String combinePaths(String classPath, String methodPath) {
        if (classPath.endsWith("/") && methodPath.startsWith("/")) {
            return classPath + methodPath.substring(1);
        }
        return classPath + methodPath;
    }

    private void addMapping(String path, String className) {
        mappings.computeIfAbsent(path, k -> new HashSet<>()).add(className);
    }

    @Override
    public void reportRisks() {
        System.out.println("Potential @RequestMapping conflict detected:");
        for (Map.Entry<String, Set<String>> entry : mappings.entrySet()) {
            if (entry.getValue().size() > 1) {
                System.out.printf("  %s\n", entry.getKey());
                for (String className : entry.getValue()) {
                    System.out.printf("    - %s\n", className);
                }
            }
        }
    }
}

