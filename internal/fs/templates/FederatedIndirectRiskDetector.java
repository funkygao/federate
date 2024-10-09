package {{.Package}};

import lombok.extern.slf4j.Slf4j;
import org.springframework.core.annotation.AnnotationUtils;
import org.springframework.web.bind.annotation.RequestMapping;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.*;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

@Slf4j
public class FederatedIndirectRiskDetector {
    private final List<RiskDetector> detectors;

    public FederatedIndirectRiskDetector() {
        this.detectors = Arrays.asList(new RequestMappingConflictDetector());
    }

    public void detectRisks() throws IOException {
        String classpath = System.getProperty("java.class.path");
        String[] classpathEntries = classpath.split(File.pathSeparator);
        log.info("Starting risk detection ...");
        for (String entry : classpathEntries) {
            if (entry.endsWith(".jar")) {
                processJar(entry);
            }
        }

        for (RiskDetector detector : detectors) {
            detector.reportRisks();
        }

        log.info("Risk detection completed.");
    }

    private void processJar(String jarPath) throws IOException {
        log.debug("Processing {}", jarPath);
        try (JarFile jarFile = new JarFile(jarPath)) {
            Enumeration<JarEntry> entries = jarFile.entries();
            URL[] urls = { new URL("jar:file:" + jarPath + "!/") };
            try (URLClassLoader cl = URLClassLoader.newInstance(urls)) {
                while (entries.hasMoreElements()) {
                    JarEntry je = entries.nextElement();
                    if (je.isDirectory() || !je.getName().endsWith(".class") || je.getName().startsWith("META-INF/versions/")) {
                        continue;
                    }
                    String className = je.getName().substring(0, je.getName().length() - 6);
                    className = className.replace('/', '.');
                    try {
                        Class<?> clazz = cl.loadClass(className);
                        for (RiskDetector detector : detectors) {
                            try {
                                detector.analyzeClass(clazz);
                            } catch (Throwable t) {
                                log.trace("Error analyzing class {} with detector {}: {}", className, detector.getClass().getSimpleName(), t.getMessage());
                            }
                        }
                    } catch (ClassNotFoundException | NoClassDefFoundError e) {
                        log.trace("{}: Skipping class {} due to: {}", jarPath, className, e.getMessage());
                    } catch (UnsupportedClassVersionError e) {
                        log.trace("{}: Skipping class {} due to unsupported version", jarPath, className);
                    } catch (Throwable t) {
                        log.trace("{}: Unexpected error processing class {}: {}", jarPath, className, t.getMessage());
                    }
                }
            }
        }
    }

    private interface RiskDetector {
        void analyzeClass(Class<?> clazz);
        void reportRisks();
    }

    @Slf4j
    private static class RequestMappingConflictDetector implements RiskDetector {
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
            System.out.println("Potential RequestMapping conflict detected:");
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
}

