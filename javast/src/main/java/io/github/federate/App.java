package io.github.federate;

import io.github.federate.visitor.*;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ParserConfiguration;
import com.google.gson.Gson;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class App {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java -jar javast.jar <command> <directory_path> [arg]");
            System.exit(1);
        }

        String command = args[0];
        String directoryPath = args[1];
        Path rootPath = Paths.get(directoryPath);

        try {
            List<Path> javaFiles = Files.walk(rootPath)
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".java"))
                    .filter(p -> !isTestFile(p))
                    .collect(Collectors.toList());

            ParserConfiguration config = new ParserConfiguration();
            config.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_8);
            StaticJavaParser.setConfiguration(config);
            FileVisitor visitor = createVisitor(command, args);

            for (Path javaFile : javaFiles) {
                try {
                    CompilationUnit cu = StaticJavaParser.parse(javaFile);
                    visitor.visit(cu, javaFile);
                } catch (IOException e) {
                    System.err.println("Error parsing file " + javaFile + ": " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("Error walking through directory: " + e.getMessage());
            System.exit(1);
        }
    }

    private static boolean isTestFile(Path path) {
        String pathStr = path.toString();
        return pathStr.contains("src" + File.separator + "test" + File.separator + "java") 
               || pathStr.endsWith("Test.java");
    }

    private static FileVisitor createVisitor(String command, String[] args) {
        // args: [command, dir, ...]
        switch (command) {
            case "replace-service":
                // @Service
                validateArgsLength(args, 3, "replace-service command requires a JSON string of old and new values");
                Map<String, String> serviceMap = new Gson().fromJson(args[2], Map.class);
                return new ServiceAnnotationTransformer(serviceMap);

            case "inject-transaction-manager":
                // @Transactional, TransactionTemplate
                validateArgsLength(args, 3, "inject-transaction-manager command requires the transaction manager name");
                return new TransactionManagerInjector(args[2]);

            default:
                System.err.println("Unknown command: " + command);
                System.exit(1);
                return null;
        }
    }

    private static void validateArgsLength(String[] args, int expectedLength, String errorMessage) {
        if (args.length < expectedLength) {
            System.err.println(errorMessage);
            System.exit(1);
        }
    }
}

