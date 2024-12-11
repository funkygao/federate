// INPUT
import org.springframework.context.annotation.ImportResource;

public class Test {
    @ImportResource({"classpath:config1.xml", "file:config2.xml"})
    void method() {}
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;

public class Test {

    @ImportResource({ "classpath:federated/testComponent/config1.xml", "federated/testComponent/file:config2.xml" })
    void method() {
    }
}
