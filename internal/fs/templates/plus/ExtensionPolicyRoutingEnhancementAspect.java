package unsafe.hack;

import com.jdwl.wms.stock.ext.domain.serial.SerialStockVerificationIdentity;
import com.jdwl.wms.stock.ext.policy.SerialStockVerificationExtPolicy;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

/**
 * 演示：如何增强扩展点的路由.
 */
@Aspect
@Component
@Slf4j
public class ExtensionPolicyRoutingEnhancementAspect {

    @Around("execution(* com.jdwl.wms.stock.ext.policy.*.extensionCode(..))")
    public Object aroundExtensionCode(ProceedingJoinPoint joinPoint) throws Throwable {
        Object[] args = joinPoint.getArgs();
        if (args.length == 0 || args[0] == null) {
            return joinPoint.proceed();
        }

        Object identity = args[0];
        String policyClassName = joinPoint.getSignature().getDeclaringTypeName();
        log.debug("Enhancing extension code routing for policy: {}", policyClassName);

        if (policyClassName.endsWith(SerialStockVerificationExtPolicy.class.getSimpleName())) {
            return handleSerialStockVerificationExtPolicy(joinPoint, (SerialStockVerificationIdentity) identity);
        }

        // 如果没有匹配的特殊处理，则调用原有的路由逻辑
        return joinPoint.proceed();
    }

    private Object handleSerialStockVerificationExtPolicy(ProceedingJoinPoint joinPoint, SerialStockVerificationIdentity identity) throws Throwable {
        // 在这里添加 SerialStockVerificationExtPolicy 的自定义路由逻辑

        // 无需增强，调用原有的路由逻辑
        return joinPoint.proceed();
    }
}
