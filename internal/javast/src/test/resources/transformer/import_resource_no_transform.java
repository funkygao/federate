// INPUT
import org.springframework.context.annotation.ImportResource;

public class Test {
    @ImportResource("/federated/testComponent/config.xml")
    void method() {}
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;

public class Test {

    @ImportResource("/federated/testComponent/config.xml")
    void method() {
    }
}
