package io.github.federate;

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
import java.util.stream.Collectors;

public class App {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java -jar javast.jar <command> <directory_path>");
            System.err.println("Available commands: parse, analyze-methods, analyze-classes");
            System.exit(1);
        }

        String command = args[0];
        String directoryPath = args[1];
        Path rootPath = Paths.get(directoryPath);

        List<ParserResult> results = new ArrayList<>();

        try {
            List<Path> javaFiles = Files.walk(rootPath)
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".java"))
                    .filter(p -> !isTestFile(p))
                    .collect(Collectors.toList());

            ParserConfiguration config = new ParserConfiguration();
            config.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_8);
            StaticJavaParser.setConfiguration(config);

            for (Path javaFile : javaFiles) {
                try {
                    CompilationUnit cu = StaticJavaParser.parse(javaFile);
                    JavaFileVisitor visitor = createVisitor(command);
                    visitor.visit(cu, null);
                    results.add(new ParserResult(rootPath.relativize(javaFile).toString(), visitor.getClasses(), visitor.getMethods()));
                } catch (IOException e) {
                    System.err.println("Error parsing file " + javaFile + ": " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("Error walking through directory: " + e.getMessage());
            System.exit(1);
        }

        Gson gson = new Gson();
        String jsonResult = gson.toJson(results);
        // stdout is golang's stdin
        System.out.println(jsonResult);
    }

    private static boolean isTestFile(Path path) {
        String pathStr = path.toString();
        return pathStr.contains("src" + File.separator + "test" + File.separator + "java") 
               || pathStr.endsWith("Test.java");
    }

    private static JavaFileVisitor createVisitor(String command) {
        switch (command) {
            case "parse":
                return new JavaFileVisitor();
            default:
                System.err.println("Unknown command: " + command);
                System.exit(1);
                return null; // This line will never be reached
        }
    }
}

