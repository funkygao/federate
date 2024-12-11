// INPUT
import javax.annotation.Resource;

public class OuterClass {
    @Resource
    private TestService testService;

    class InnerClass {
        @Resource
        private TestService testService;
    }
}

// EXPECTED
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class OuterClass {

    @Autowired
    private TestService testService;

    class InnerClass {

        @Autowired
        private TestService testService;
    }
}
