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
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class App {
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.err.println("Usage: java -jar javast.jar <directory_path> [<command> <args>]...");
            System.exit(1);
        }

        List<Path> javaFiles = Files.walk(Paths.get(args[0]))
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".java"))
                .filter(p -> !isTestFile(p))
                .collect(Collectors.toList());

        ParserConfiguration config = new ParserConfiguration();
        config.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_8);
        StaticJavaParser.setConfiguration(config);

        CompositeVisitor compositeVisitor = new CompositeVisitor();
        for (int i = 1; i < args.length; i += 2) {
            String commandName = args[i];
            String commandArg = (i + 1 < args.length) ? args[i + 1] : "";
            BaseCodeModifier visitor = createVisitor(commandName, commandArg);
            if (visitor != null) {
                compositeVisitor.addVisitor(visitor);
            }
        }

        for (Path javaFile : javaFiles) {
            CompilationUnit cu = StaticJavaParser.parse(javaFile);
            compositeVisitor.visit(cu, javaFile);
        }
    }

    private static BaseCodeModifier createVisitor(String command, String cmdSpecificArg) {
        switch (command) {
            case "replace-service":
                Map<String, String> serviceMap = new Gson().fromJson(cmdSpecificArg, Map.class);
                return new TransformerServiceName(serviceMap);

            case "inject-transaction-manager":
                String trxManager = cmdSpecificArg;
                return new TransformerTrxManager(trxManager);

            case "update-property-keys":
                Map<String, String> keyMapping = new Gson().fromJson(cmdSpecificArg, Map.class);
                return new TransformerPropertyKeyRef(keyMapping);

            case "transform-import-resource":
                String componentName = cmdSpecificArg;
                return new TransformerImportResource(componentName);

            case "transform-resource":
                return new TransformerResourceToAutowired();

            default:
                System.err.println("Unknown command: " + command);
                System.exit(1);
                return null;
        }
    }

    private static boolean isTestFile(Path path) {
        String pathStr = path.toString();
        return pathStr.contains("src" + File.separator + "test" + File.separator + "java")
               || pathStr.endsWith("Test.java");
    }
}

