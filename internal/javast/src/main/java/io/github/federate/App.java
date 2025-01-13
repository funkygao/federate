package io.github.federate;

import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.Gson;
import io.github.federate.extractor.ASTExtractorVisitor;
import io.github.federate.extractor.BaseExtractor;
import io.github.federate.visitor.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitor;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

public class App {
    private static final List<String> IGNORED_DIRECTORIES = Arrays.asList(
            ".git",
            ".idea",
            ".gradle",
            ".mvn",
            "target",
            "build",
            "out",
            "bin"
    );

    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: java -jar javast.jar <directory_path> [<command> <args>]...");
            System.exit(1);
        }

        List<Path> javaFiles = new ArrayList<>();
        Files.walkFileTree(Paths.get(args[0]), new FileVisitor<Path>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                if (isIgnoredPath(dir)) {
                    return FileVisitResult.SKIP_SUBTREE;
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                if (file.toString().endsWith(".java") && !isTestFile(file)) {
                    javaFiles.add(file);
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) {
                return FileVisitResult.CONTINUE;
            }
        });

        ParserConfiguration config = new ParserConfiguration();
        config.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_8);
        StaticJavaParser.setConfiguration(config);

        List<BaseExtractor> extractors = new LinkedList<>();
        CompositeVisitor compositeVisitor = new CompositeVisitor();
        for (int i = 1; i < args.length; i += 2) {
            String commandName = args[i];
            String commandArg = (i + 1 < args.length) ? args[i + 1] : "";
            BaseCodeModifier visitor = createVisitor(commandName, commandArg);
            if (visitor != null) {
                compositeVisitor.addVisitor(visitor);
            }

            BaseExtractor voidVisitor = createExtractor(commandName, commandArg);
            if (voidVisitor != null) {
                extractors.add(voidVisitor);
            }
        }

        for (Path javaFile : javaFiles) {
            CompilationUnit cu = StaticJavaParser.parse(javaFile);
            compositeVisitor.visit(cu, javaFile);

            for (BaseExtractor extractor : extractors) {
                extractor.visit(cu, null);
            }
        }

        for (BaseExtractor extractor : extractors) {
            extractor.finish();
        }
    }

    private static BaseExtractor createExtractor(String command, String cmdSpecificArg) {
        switch (command) {
            case "extract-ast":
                return new ASTExtractorVisitor();
            default:
                return null;
        }

    }

    private static BaseCodeModifier createVisitor(String command, String cmdSpecificArg) {
        switch (command) {
            case "replace-service":
                Map<String, String> serviceMap = new Gson().fromJson(cmdSpecificArg, Map.class);
                return new TransformerServiceName(serviceMap);

            case "inject-transaction-manager":
                return new TransformerTrxManager(cmdSpecificArg);

            case "update-property-keys":
                Map<String, String> keyMapping = new Gson().fromJson(cmdSpecificArg, Map.class);
                return new TransformerPropertyKeyRef(keyMapping);

            case "transform-import-resource":
                return new TransformerImportResource(cmdSpecificArg);

            case "transform-resource":
                return new TransformerResourceToAutowired();

            default:
                return null;
        }
    }

    private static boolean isIgnoredPath(Path path) {
        String normalizedPath = path.normalize().toString().replace(File.separator, "/");
        for (String ignoredDir : IGNORED_DIRECTORIES) {
            if (normalizedPath.contains("/" + ignoredDir + "/")) {
                return true;
            }
        }
        return false;
    }

    private static boolean isTestFile(Path path) {
        String pathStr = path.toString();
        return pathStr.contains("src" + File.separator + "test" + File.separator + "java")
                || pathStr.endsWith("Test.java");
    }
}

