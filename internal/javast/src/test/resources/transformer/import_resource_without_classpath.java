// INPUT
import org.springframework.context.annotation.ImportResource;

public class Test {
    @ImportResource("/foo/config.xml")
    void method() {}
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;

public class Test {

    @ImportResource("/federated/testComponent/foo/config.xml")
    void method() {
    }
}
