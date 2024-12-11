// INPUT
import javax.annotation.Resource;

public class TestClass {
    @Resource
    private TestService testService;
    
    @Resource(name = "customName")
    private TestService anotherTestService;
    
    @Resource
    public void setTestDao(TestDao testDao) {
        // setter method
    }
}

// EXPECTED
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {

    @Autowired
    @Qualifier("testService")
    private TestService testService;

    @Autowired
    @Qualifier("customName")
    private TestService anotherTestService;

    @Autowired
    @Qualifier("testDao")
    public void setTestDao(TestDao testDao) {
        // setter method
    }
}
