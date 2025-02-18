package io.github.federate.or;

import com.graphhopper.jsprit.core.algorithm.VehicleRoutingAlgorithm;
import com.graphhopper.jsprit.core.algorithm.box.Jsprit;
import com.graphhopper.jsprit.core.problem.Location;
import com.graphhopper.jsprit.core.problem.VehicleRoutingProblem;
import com.graphhopper.jsprit.core.problem.job.Service;
import com.graphhopper.jsprit.core.problem.solution.VehicleRoutingProblemSolution;
import com.graphhopper.jsprit.core.problem.vehicle.VehicleImpl;
import com.graphhopper.jsprit.core.problem.vehicle.VehicleTypeImpl;
import com.graphhopper.jsprit.core.reporting.SolutionPrinter;
import com.graphhopper.jsprit.core.util.Solutions;
import org.junit.jupiter.api.Test;

import java.util.Collection;

class OperationResearchTest {

    @Test
    void VRP() {
        // 创建一个车辆路径问题（VRP）的构建器
        // VRP 是一个优化问题，目标是找到一组最优的路径来服务所有客户
        VehicleRoutingProblem.Builder vrpBuilder = VehicleRoutingProblem.Builder.newInstance();

        // 定义车辆类型
        // 在这个例子中，我们定义了一种可以携带2个单位货物的车辆类型
        VehicleTypeImpl.Builder vehicleTypeBuilder = VehicleTypeImpl.Builder.newInstance("vehicleType")
                .addCapacityDimension(0, 2);
        VehicleTypeImpl vehicleType = vehicleTypeBuilder.build();

        // 创建一个具体的车辆实例
        // 这辆车从坐标(0,0)出发，使用我们刚刚定义的车辆类型
        VehicleImpl.Builder vehicleBuilder = VehicleImpl.Builder.newInstance("vehicle");
        vehicleBuilder.setStartLocation(Location.newInstance(0, 0));
        vehicleBuilder.setType(vehicleType);
        VehicleImpl vehicle = vehicleBuilder.build();

        // 将车辆添加到问题中
        vrpBuilder.addVehicle(vehicle);

        // 添加服务点（即客户需求）
        // 我们随机生成5个服务点，每个服务点需要1个单位的货物
        for (int i = 0; i < 5; i++) {
            Service service = Service.Builder.newInstance("" + i)
                    .addSizeDimension(0, 1)
                    .setLocation(Location.newInstance(Math.random() * 100, Math.random() * 100))
                    .build();
            vrpBuilder.addJob(service);
        }

        // 构建完整的车辆路径问题
        VehicleRoutingProblem problem = vrpBuilder.build();

        // 创建求解算法
        // Jsprit 是一个启发式算法，用于解决车辆路径问题
        VehicleRoutingAlgorithm algorithm = Jsprit.createAlgorithm(problem);

        // 运行算法来寻找解决方案
        Collection<VehicleRoutingProblemSolution> solutions = algorithm.searchSolutions();

        // 从所有解决方案中选择最佳的一个
        VehicleRoutingProblemSolution bestSolution = Solutions.bestOf(solutions);

        // 打印结果
        // 这将显示最佳路径、总距离、每个服务点的访问顺序等信息
        SolutionPrinter.print(problem, bestSolution, SolutionPrinter.Print.VERBOSE);
    }
}
