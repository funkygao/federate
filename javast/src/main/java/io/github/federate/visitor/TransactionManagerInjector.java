package io.github.federate.visitor;

import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.Visitable;

// 显式注入事务管理器：如果现有代码未指定
public class TransactionManagerInjector extends BaseCodeModifier {
    private final String transactionManagerName;

    public TransactionManagerInjector(String transactionManagerName) {
        this.transactionManagerName = transactionManagerName;
    }

    // 处理无参数注解 @Transactional
    @Override
    public Visitable visit(MarkerAnnotationExpr n, Void arg) {
        if ("Transactional".equals(n.getNameAsString())) {
            // 将无参数注解转换为有参数注解，并添加 transactionManager
            NormalAnnotationExpr nae = new NormalAnnotationExpr();
            nae.setName(n.getName());
            nae.addPair("transactionManager", new StringLiteralExpr(transactionManagerName));
            modified = true;
            return nae;
        }
        return super.visit(n, arg);
    }

    // 处理有参数注解 @Transactional(...)
    @Override
    public Visitable visit(NormalAnnotationExpr n, Void arg) {
        if ("Transactional".equals(n.getNameAsString())) {
            boolean hasTransactionManager = n.getPairs().stream()
                    .anyMatch(pair -> "transactionManager".equals(pair.getNameAsString()));
            if (!hasTransactionManager) {
                // 添加 transactionManager 属性
                n.addPair("transactionManager", new StringLiteralExpr(transactionManagerName));
                modified = true;
            }
        }
        return super.visit(n, arg);
    }

    // 处理单参数注解 @Transactional("value")
    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        if ("Transactional".equals(n.getNameAsString())) {
            // 将单参数注解转换为有参数注解，并添加 transactionManager
            NormalAnnotationExpr nae = new NormalAnnotationExpr();
            nae.setName(n.getName());
            nae.addPair("value", n.getMemberValue());
            nae.addPair("transactionManager", new StringLiteralExpr(transactionManagerName));
            modified = true;
            return nae;
        }
        return super.visit(n, arg);
    }

    // 处理编程式事务管理
    @Override
    public Visitable visit(ObjectCreationExpr n, Void arg) {
        String typeName = n.getType().getNameAsString();
        if ("TransactionTemplate".equals(typeName) || "DefaultTransactionDefinition".equals(typeName)) {
            if (n.getArguments().isEmpty()) {
                // 添加 transactionManager 变量作为构造函数参数
                n.addArgument(new NameExpr(transactionManagerName));
                modified = true;
            }
        }
        return super.visit(n, arg);
    }
}

