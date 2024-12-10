package io.github.federate.visitor;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.Visitable;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Transformer to update property keys in annotations.
 *
 * <p>根据 {@code keyMapping} 规则，oldKey -> newKey 改写Java源代码.</p>
 * <p>例如：`@Value("#{'${foo}'.split(',')}")`，keyMapping 里包含：foo -> egg.foo，则替换为：`@Value("#{'${egg.foo}'.split(',')}")`</p>
 */
public class PropertyKeyTransformer extends BaseCodeModifier {
    private final Set<String> annotationsToProcess = new HashSet<>(Arrays.asList(
            "Value", "LafValue", "ConfigurationProperties", "ConditionalOnProperty",
            "RequestMapping", "GetMapping", "PostMapping", "PutMapping", "DeleteMapping"
    ));
    private final Map<String, String> keyMapping;
    private final Pattern placeholderPattern = Pattern.compile("\\$\\{([^}]+)}");
    private final Pattern directKeyPattern;

    public PropertyKeyTransformer(Map<String, String> keyMapping) {
        this.keyMapping = keyMapping;
        this.directKeyPattern = Pattern.compile("\\b(" + String.join("|", keyMapping.keySet()) + ")\\b");
    }

    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        transformAnnotation(n);
        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(NormalAnnotationExpr n, Void arg) {
        transformAnnotation(n);
        return super.visit(n, arg);
    }

    void transformAnnotation(AnnotationExpr n) {
        if (!annotationsToProcess.contains(n.getNameAsString())) {
            return;
        }

        if (n instanceof SingleMemberAnnotationExpr) {
            // 单个参数
            // @Value("${foo}")
            SingleMemberAnnotationExpr smae = (SingleMemberAnnotationExpr) n;
            Expression memberValue = smae.getMemberValue();
            Expression newMemberValue = transformAnnotationExpression(memberValue);
            if (newMemberValue != memberValue) {
                // key reference transformed
                smae.setMemberValue(newMemberValue);
                modified = true;
            }
        } else if (n instanceof NormalAnnotationExpr) {
            // 多个参数
            // @ConfigurationProperties(prefix = "foo")
            // @ConditionalOnProperty(name = "foo", havingValue = "true")
            NormalAnnotationExpr nae = (NormalAnnotationExpr) n;
            for (MemberValuePair pair : nae.getPairs()) {
                Expression oldValue = pair.getValue();
                Expression newValue = transformAnnotationExpression(oldValue);
                if (newValue != oldValue) {
                    // key reference transformed
                    pair.setValue(newValue);
                    modified = true;
                }
            }
        } else if (n instanceof MarkerAnnotationExpr) {
            // @Autowired，这里不处理
        }
    }

    Expression transformAnnotationExpression(Expression expr) {
        if (expr instanceof StringLiteralExpr) {
            StringLiteralExpr sle = (StringLiteralExpr) expr;
            String newValue = replaceKeys(sle.getValue());
            if (!newValue.equals(sle.getValue())) {
                return new StringLiteralExpr(newValue);
            }
        } else if (expr instanceof ArrayInitializerExpr) {
            // @PostMapping({"${schedule.clover.webAlias}/checkTaskTooManyAlarm"})
            ArrayInitializerExpr aie = (ArrayInitializerExpr) expr;
            NodeList<Expression> values = new NodeList<>();
            boolean changed = false;
            for (Expression e : aie.getValues()) {
                Expression newExpr = transformAnnotationExpression(e);
                if (newExpr != e) {
                    changed = true;
                }
                values.add(newExpr);
            }
            if (changed) {
                return new ArrayInitializerExpr(values);
            }
        }
        return expr;
    }

    String replaceKeys(String str) {
        String result = replacePlaceholderKeys(str);
        return replaceDirectKeys(result);
    }

    private String replacePlaceholderKeys(String str) {
        StringBuffer sb = new StringBuffer();
        Matcher matcher = placeholderPattern.matcher(str);
        while (matcher.find()) {
            String key = matcher.group(1);
            String newKey = keyMapping.getOrDefault(key, key);
            matcher.appendReplacement(sb, Matcher.quoteReplacement("${" + newKey + "}"));
        }
        matcher.appendTail(sb);
        return sb.toString();
    }

    private String replaceDirectKeys(String str) {
        StringBuffer sb = new StringBuffer();
        Matcher matcher = directKeyPattern.matcher(str);
        while (matcher.find()) {
            String key = matcher.group(1);
            String newKey = keyMapping.get(key);
            if (newKey != null && !isWithinPlaceholder(str, matcher.start())) {
                matcher.appendReplacement(sb, Matcher.quoteReplacement(newKey));
            } else {
                matcher.appendReplacement(sb, Matcher.quoteReplacement(key));
            }
        }
        matcher.appendTail(sb);
        return sb.toString();
    }

    private boolean isWithinPlaceholder(String str, int index) {
        int openBrace = str.lastIndexOf("${", index);
        int closeBrace = str.lastIndexOf("}", index);
        return openBrace > closeBrace;
    }
}
