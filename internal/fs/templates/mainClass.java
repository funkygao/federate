package {{.Package}};

import com.jdwl.wms.common.federation.*;
import org.springframework.boot.Banner;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.*;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * Main entry point.
 */
@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class})
@PropertySources(value = {
        @PropertySource(value = {"/federated/federated.properties"}, encoding = "utf-8"),
        @PropertySource(value = {"/federated/application.yml"}, encoding = "utf-8"),
})
@ImportResource(locations = {"/federated/spring.xml"})
@Import({
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
public class {{.ClassName}} {
    public static void main(String[] args) {
        excludeBeans();

        SpringApplicationBuilder builder = new SpringApplicationBuilder({{.ClassName}}.class)
                .profiles("{{.Profile}}")
                .bannerMode(Banner.Mode.OFF)
                .properties("spring.config.name=none") // 禁用默认的 application.yml 加载
                .properties("spring.config.location=/federated/application.yml")
                .properties("spring.config.additional-location=/federated/application.yml")
                .properties("spring.main.allow-bean-definition-overriding=true")
                .lazyInitialization(true)
                .beanNameGenerator(new FederatedDefaultBeanNameGenerator())
                .resourceLoader(new FederatedResourceLoader({{.ClassName}}.class.getClassLoader()))
                .logStartupInfo(false);
        builder.run(args);
    }

    private static void excludeBeans() {
        {{- range .ExcludedTypes }}
        FederatedExcludedTypeFilter.exclude({{.}}.class);
        {{- end }}
    }
}
