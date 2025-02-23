// Code generated by federate, DO NOT EDIT.
package {{.Package}};

import {{.FederatedRuntimePackage}}.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.Banner;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.*;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.transaction.annotation.EnableTransactionManagement;

import java.util.Arrays;

/**
 * Main entry point.
 */
@SpringBootApplication(exclude = {
    {{- range .Excludes}}
        {{.}}.class,
    {{- end}}
})
@PropertySources(value = {
        @PropertySource(value = {"/federated/application.properties"}, encoding = "utf-8",
                factory = com.jd.security.configsec.spring.config.JDSecurityPropertySourceFactory.class),
        @PropertySource(value = {"/federated/application.yml"}, encoding = "utf-8"),
})
@ImportResource(locations = {"/federated/spring.xml"})
@Import({
        //FederatedMybatisConfig.class,
    {{- range .Imports}}
        {{.}}.class,
    {{- end}}
})
@ComponentScan(
        basePackages = {
        {{- range .BasePackages}}
            "{{.}}",
        {{- end}}
        },
        nameGenerator = FederatedAnnotationBeanNameGenerator.class,
        excludeFilters = @ComponentScan.Filter(type = FilterType.CUSTOM, classes = FederatedExcludedTypeFilter.class)
)
@EnableAspectJAutoProxy(exposeProxy = true, proxyTargetClass = true)
@EnableTransactionManagement
@EnableAsync
@Slf4j
public class {{.ClassName}} {

    public static void main(String[] args) throws Exception {
        if (Arrays.asList(args).contains("--detect-indirect-risk")) {
            new FederatedIndirectRiskDetector().detectRisks();
        } else {
            startSpringBoot(args);
        }
    }

    private static void startSpringBoot(String[] args) {
        excludeBeans();

        SpringApplicationBuilder builder = new SpringApplicationBuilder({{.ClassName}}.class)
                .profiles("{{.Profile}}")
                .bannerMode(Banner.Mode.OFF)
                .properties("spring.config.name=none") // 禁用默认的 application.yml 加载
                .properties("spring.config.location=/federated/application.yml")
                .properties("spring.config.additional-location=/federated/application.yml")
                .properties("spring.main.allow-bean-definition-overriding=true")
                .lazyInitialization(true)
                .beanNameGenerator(new FederatedAnnotationBeanNameGenerator())
                .resourceLoader(new FederatedResourceLoader({{.ClassName}}.class.getClassLoader()))
                .logStartupInfo(false);
        ApplicationContext context = builder.run(args);
        log.info("FederatedApplication: {{.App}}, started");
    }

    private static void excludeBeans() {
        {{- range .ExcludedTypes }}
        FederatedExcludedTypeFilter.exclude({{.}}.class);
        {{- end }}
    }
}
