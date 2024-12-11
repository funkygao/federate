// INPUT
import org.springframework.context.annotation.ImportResource;

public class Test {
    @ImportResource({"config1.xml", "config2.xml"})
    void method() {}
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;

public class Test {

    @ImportResource({ "federated/testComponent/config1.xml", "federated/testComponent/config2.xml" })
    void method() {
    }
}
