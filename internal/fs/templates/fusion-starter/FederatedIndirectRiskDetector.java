package {{.Package}};

import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

@Slf4j
public class FederatedIndirectRiskDetector {
    private final List<RiskDetector> detectors;

    public FederatedIndirectRiskDetector() {
        detectors = new LinkedList<>();
        // builtin
        detectors.add(new RequestMappingConflictDetector());
        // addon
        {{- range .AddOns }}
        detectors.add(new {{.}}());
        {{- end }}
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

}

