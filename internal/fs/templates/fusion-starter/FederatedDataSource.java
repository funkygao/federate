// Code generated by federate, DO NOT EDIT.
package {{.Package}};

import com.jdwl.wms.common.datasource.RoutingDataSourceFactoryBean;
import com.jdwl.wms.common.datasource.RoutingRuleProvider;
import com.jdwl.wms.common.federation.FederatedAnnotationBeanNameGenerator;
import com.jdwl.wms.rule.api.RuleMatchQueryService;
import com.zaxxer.hikari.HikariDataSource;
import org.apache.ibatis.session.SqlSessionFactory;
import org.mybatis.spring.SqlSessionFactoryBean;
import org.mybatis.spring.annotation.MapperScan;
import org.mybatis.spring.annotation.MapperScans;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.DependsOn;
import org.springframework.context.annotation.Primary;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.support.SQLErrorCodesFactory;
import org.springframework.transaction.PlatformTransactionManager;

import javax.sql.DataSource;
import java.util.HashMap;
import java.util.Map;

public class FederatedDataSource {
    private static final String CONFIG = "mybatis-config.xml";
    private static final String MAPPERS = "classpath*:META-INF/mappers/*.xml";
}
