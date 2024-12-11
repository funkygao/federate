// INPUT
import javax.annotation.Resource;

public class TestClass {
    @Resource
    public void setTestService(TestService service) {
    }

    @Resource(name = "special")
    public void setAnotherService(TestService service) {
    }
}

// EXPECTED
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {

    @Autowired
    @Qualifier("testService")
    public void setTestService(TestService service) {
    }

    @Autowired
    @Qualifier("special")
    public void setAnotherService(TestService service) {
    }
}
