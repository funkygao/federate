// INPUT
import org.springframework.context.annotation.ImportResource;
import org.springframework.context.annotation.PropertySource;

@ImportResource({"classpath:applicationContext-common.xml", "classpath:applicationContext-beans.xml"})
@PropertySource("classpath:foo.properties")
public class AppConfig {
    // Some code here
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;
import org.springframework.context.annotation.PropertySource;

@ImportResource({ "classpath:federated/testComponent/applicationContext-common.xml", "classpath:federated/testComponent/applicationContext-beans.xml" })
@PropertySource("classpath:foo.properties")
public class AppConfig {
    // Some code here
}
