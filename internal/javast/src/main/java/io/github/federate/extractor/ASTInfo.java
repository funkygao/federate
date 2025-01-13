package io.github.federate.extractor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ASTInfo {
    List<String> imports = new ArrayList<>();
    List<String> classes = new ArrayList<>();
    List<String> methods = new ArrayList<>();
    List<String> variables = new ArrayList<>();
    List<String> variableReferences = new ArrayList<>();
    List<String> methodCalls = new ArrayList<>();

    Map<String, List<String>> inheritance = new HashMap<>();
    Map<String, List<String>> interfaces = new HashMap<>();
    List<String> annotations = new ArrayList<>();

    List<ComplexCondition> complexConditions = new ArrayList<>();
}
