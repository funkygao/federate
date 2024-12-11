// INPUT
import org.springframework.context.annotation.ImportResource;

public class Test {
    @ImportResource("classpath:config.xml")
    void method() {}
}

// EXPECTED
import org.springframework.context.annotation.ImportResource;

public class Test {

    @ImportResource("classpath:federated/testComponent/config.xml")
    void method() {
    }
}
