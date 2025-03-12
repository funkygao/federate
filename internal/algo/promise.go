package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const (
	InitialMinRadius = 3.0 // 最小半径
	InitialMaxRadius = 8.0 // 最大半径
)

type PromiseSystem struct {
	ActiveOrders     int
	AvailableRiders  int
	CurrentRadius    float64
	TrafficIndex     float64
	WeatherCondition float64
	TimeOfDay        int
	BaseRadius       float64
	maxRadius        float64
	minRadius        float64
}

func NewPromiseSystem() *PromiseSystem {
	return &PromiseSystem{
		ActiveOrders:     100,
		AvailableRiders:  20,
		CurrentRadius:    5.0,
		TrafficIndex:     0.3,
		WeatherCondition: 0.1,
		TimeOfDay:        9,
		BaseRadius:       5.0,
		maxRadius:        InitialMaxRadius,
		minRadius:        InitialMinRadius,
	}
}

func (ps *PromiseSystem) calculateAverageRiderWorkload() float64 {
	averageOrdersPerRider := float64(ps.ActiveOrders) / float64(ps.AvailableRiders)
	averageHoursWorked := 4.0 + float64(ps.TimeOfDay%8)*0.5
	return (averageOrdersPerRider + averageHoursWorked) / 2
}

func (ps *PromiseSystem) adjustRadius() {
	pressure := ps.calculatePressure()
	optimalPressure := 2.0
	adjustment := (optimalPressure - pressure) * 0.4 // 增加调整幅度

	// 当压力很低时，更快地缩小半径
	if pressure < 1.0 {
		adjustment -= (1.0 - pressure) * 0.8
	}

	newRadius := ps.CurrentRadius + adjustment
	ps.CurrentRadius = math.Max(ps.minRadius, math.Min(ps.maxRadius, newRadius))
}

func (ps *PromiseSystem) updateSystemStatus() {
	ps.TimeOfDay = (ps.TimeOfDay + 1) % 24

	// 更动态地调整订单数
	baseOrderChange := rand.Float64()*0.3 - 0.15
	timeBasedOrderFactor := 1.0
	if ps.TimeOfDay >= 11 && ps.TimeOfDay <= 13 {
		timeBasedOrderFactor = 1.5 // 增加午餐高峰
	} else if ps.TimeOfDay >= 17 && ps.TimeOfDay <= 19 {
		timeBasedOrderFactor = 1.8 // 增加晚餐高峰
	} else if ps.TimeOfDay >= 22 || ps.TimeOfDay <= 6 {
		timeBasedOrderFactor = 0.3 // 降低夜间订单
	}
	orderChange := baseOrderChange * timeBasedOrderFactor

	// 更平衡地调整骑手数
	riderChange := (rand.Float64()*0.1 - 0.05) * (float64(ps.ActiveOrders) / float64(ps.AvailableRiders) / 2)

	ps.ActiveOrders = int(float64(ps.ActiveOrders) * (1 + orderChange))
	ps.AvailableRiders = int(float64(ps.AvailableRiders) * (1 + riderChange))

	// 确保最小值和最大值
	ps.ActiveOrders = int(math.Max(math.Min(float64(ps.ActiveOrders), 200), 30))
	ps.AvailableRiders = int(math.Max(math.Min(float64(ps.AvailableRiders), 30), 15))

	ps.TrafficIndex = math.Max(0, math.Min(1, ps.TrafficIndex+(rand.Float64()-0.5)*0.1))
	ps.WeatherCondition = math.Max(0, math.Min(1, ps.WeatherCondition+(rand.Float64()-0.5)*0.05))

	ps.adjustRadius()
}

func (ps *PromiseSystem) calculatePressure() float64 {
	baseLoad := float64(ps.ActiveOrders) / float64(ps.AvailableRiders)

	trafficFactor := 1 + ps.TrafficIndex*0.4
	weatherFactor := 1 + ps.WeatherCondition*0.3

	timeFactor := 1.0
	if (ps.TimeOfDay >= 11 && ps.TimeOfDay <= 13) || (ps.TimeOfDay >= 17 && ps.TimeOfDay <= 19) {
		timeFactor = 1.3
	} else if ps.TimeOfDay >= 23 || ps.TimeOfDay <= 5 {
		timeFactor = 0.9 // 增加夜间压力
	}

	averageWorkload := ps.calculateAverageRiderWorkload()
	workloadFactor := math.Max(1, averageWorkload/8) * 0.4

	pressure := baseLoad * trafficFactor * weatherFactor * timeFactor * workloadFactor

	return math.Min(pressure, 4)
}

func (ps *PromiseSystem) ETA() float64 {
	basetime := 20.0                             // 基础时间，单位分钟
	pressureFactor := ps.calculatePressure() * 6 // 增加压力影响
	distanceFactor := ps.CurrentRadius * 2.5     // 增加距离影响
	weatherFactor := ps.WeatherCondition * 15    // 增加天气影响
	trafficFactor := ps.TrafficIndex * 20        // 增加交通影响

	return basetime + pressureFactor + distanceFactor + weatherFactor + trafficFactor
}

func formatOutput(timeOfDay, activeOrders, availableRiders int, pressure, radius, promiseTime float64) string {
	return fmt.Sprintf("%02d:00 | 订单:%3d | 骑手:%2d | 压力:%.2f | 半径:%5.2fkm | 承诺:%3.0f分钟",
		timeOfDay, activeOrders, availableRiders, pressure, radius, promiseTime)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	ps := NewPromiseSystem()

	// 打印初始状态
	fmt.Println(formatOutput(ps.TimeOfDay, ps.ActiveOrders, ps.AvailableRiders,
		ps.calculatePressure(), ps.CurrentRadius, ps.ETA()))

	for i := 0; i < 48; i++ {
		ps.updateSystemStatus()
		fmt.Println(formatOutput(ps.TimeOfDay, ps.ActiveOrders, ps.AvailableRiders,
			ps.calculatePressure(), ps.CurrentRadius, ps.ETA()))
	}
}
