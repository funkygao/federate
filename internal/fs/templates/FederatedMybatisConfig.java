package {{.Package}};

import lombok.extern.slf4j.Slf4j;
import org.mybatis.spring.mapper.MapperScannerConfigurer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

@Configuration
@Slf4j
public class FederatedMybatisConfig {
    @Bean
    @Primary
    public MapperScannerConfigurer mapperScannerConfigurer() {
        log.info("Configured");
        MapperScannerConfigurer scannerConfigurer = new MapperScannerConfigurer();
        scannerConfigurer.setBasePackage({{.BasePackage}});
        scannerConfigurer.setNameGenerator(new FederatedAnnotationBeanNameGenerator());
        return scannerConfigurer;
    }

}

