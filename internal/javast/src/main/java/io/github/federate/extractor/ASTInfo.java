package io.github.federate.extractor;

import java.util.ArrayList;
import java.util.List;

public class ASTInfo {
    List<String> imports = new ArrayList<>();
    List<String> classes = new ArrayList<>();
    List<String> methods = new ArrayList<>();
    List<String> variables = new ArrayList<>();
    List<String> methodCalls = new ArrayList<>();
}
