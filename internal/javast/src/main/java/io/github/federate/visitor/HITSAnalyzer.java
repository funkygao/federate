package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class HITSAnalyzer extends VoidVisitorAdapter<Void> implements FileVisitor {
    private Map<String, Set<String>> dependencies = new HashMap<>();
    private Map<String, Double> hubScores = new HashMap<>();
    private Map<String, Double> authorityScores = new HashMap<>();
    private String currentFile;

    @Override
    public void visit(CompilationUnit cu, Path filePath) {
        currentFile = filePath.toString();
        dependencies.put(currentFile, new HashSet<>());
        hubScores.put(currentFile, 1.0);
        authorityScores.put(currentFile, 1.0);
        super.visit(cu, null);
    }

    @Override
    public void visit(ImportDeclaration n, Void arg) {
        String importName = n.getNameAsString();
        dependencies.get(currentFile).add(importName);
        super.visit(n, arg);
    }

    public void runHITS(int iterations) {
        for (int i = 0; i < iterations; i++) {
            // Update authority scores
            Map<String, Double> newAuthorityScores = new HashMap<>();
            for (String file : dependencies.keySet()) {
                double authorityScore = 0;
                for (Map.Entry<String, Set<String>> entry : dependencies.entrySet()) {
                    if (entry.getValue().contains(file)) {
                        authorityScore += hubScores.get(entry.getKey());
                    }
                }
                newAuthorityScores.put(file, authorityScore);
            }

            // Update hub scores
            Map<String, Double> newHubScores = new HashMap<>();
            for (String file : dependencies.keySet()) {
                double hubScore = 0;
                for (String dependency : dependencies.get(file)) {
                    hubScore += newAuthorityScores.getOrDefault(dependency, 0.0);
                }
                newHubScores.put(file, hubScore);
            }

            // Normalize scores
            normalizeScores(newHubScores);
            normalizeScores(newAuthorityScores);

            hubScores = newHubScores;
            authorityScores = newAuthorityScores;
        }
    }

    private void normalizeScores(Map<String, Double> scores) {
        double sum = scores.values().stream().mapToDouble(Double::doubleValue).sum();
        if (sum > 0) {
            for (Map.Entry<String, Double> entry : scores.entrySet()) {
                scores.put(entry.getKey(), entry.getValue() / sum);
            }
        } else {
            // If sum is 0, set all scores to 1/n
            double defaultScore = 1.0 / scores.size();
            for (String key : scores.keySet()) {
                scores.put(key, defaultScore);
            }
        }
    }

    public void printTopResults(int limit) {
        List<Map.Entry<String, Double>> sortedHubScores = hubScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(limit)
                .collect(Collectors.toList());

        List<Map.Entry<String, Double>> sortedAuthorityScores = authorityScores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(limit)
                .collect(Collectors.toList());

        System.out.println("Top Hub Scores:");
        for (int i = 0; i < sortedHubScores.size(); i++) {
            Map.Entry<String, Double> entry = sortedHubScores.get(i);
            System.out.printf("%d. %s (Score: %.6f)%n", i + 1, entry.getKey(), entry.getValue());
        }

        System.out.println("\nTop Authority Scores:");
        for (int i = 0; i < sortedAuthorityScores.size(); i++) {
            Map.Entry<String, Double> entry = sortedAuthorityScores.get(i);
            System.out.printf("%d. %s (Score: %.6f)%n", i + 1, entry.getKey(), entry.getValue());
        }
    }
}

