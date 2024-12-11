// INPUT
import org.springframework.context.annotation.ImportResource;

public class Test {
    @ImportResource(locations = {"classpath:config1.xml", "classpath:config2.xml"})
    void method() {}
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;

public class Test {

    @ImportResource(locations = { "classpath:federated/testComponent/config1.xml", "classpath:federated/testComponent/config2.xml" })
    void method() {
    }
}
