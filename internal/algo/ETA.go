package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Order struct {
	ID               string
	TotalItems       int
	CompletedItems   int
	StartTime        time.Time
	EstimatedTime    time.Duration
	ActualTimes      []time.Duration
	TrafficFactor    float64
	WeatherFactor    float64
	ComplexityFactor float64
	LastUpdateTime   time.Time
}

func NewOrder(id string, totalItems int) *Order {
	return &Order{
		ID:               id,
		TotalItems:       totalItems,
		StartTime:        time.Now(),
		EstimatedTime:    estimateInitialTime(totalItems),
		TrafficFactor:    1.0,
		WeatherFactor:    1.0,
		ComplexityFactor: calculateComplexity(totalItems),
		LastUpdateTime:   time.Now(),
	}
}

// 估算初始完成时间
func estimateInitialTime(items int) time.Duration {
	baseTime := 15 * time.Minute
	return baseTime + time.Duration(items*2)*time.Minute
}

// 计算订单复杂度
func calculateComplexity(items int) float64 {
	return 1.0 + math.Log(float64(items))/10
}

// UpdateProgress 更新订单进度
func (o *Order) UpdateProgress(completedItems int) {
	now := time.Now()
	o.CompletedItems = completedItems
	if o.CompletedItems > o.TotalItems {
		o.CompletedItems = o.TotalItems
	}

	// 记录实际耗时
	timeTaken := now.Sub(o.LastUpdateTime)
	o.ActualTimes = append(o.ActualTimes, timeTaken)

	// 更新交通和天气因素
	o.updateExternalFactors()

	o.LastUpdateTime = now
}

// 更新外部因素
func (o *Order) updateExternalFactors() {
	// 模拟交通状况变化
	o.TrafficFactor = 0.8 + rand.Float64()*0.4

	// 模拟天气影响
	o.WeatherFactor = 0.9 + rand.Float64()*0.2
}

// CalculateETA 计算预计到达时间
func (o *Order) CalculateETA() time.Duration {
	if o.CompletedItems == o.TotalItems {
		return 0
	}

	// 计算平均完成时间
	var totalTime time.Duration
	for _, t := range o.ActualTimes {
		totalTime += t
	}
	avgTime := totalTime / time.Duration(len(o.ActualTimes))

	// 考虑剩余项目、复杂度、交通和天气因素
	remainingItems := o.TotalItems - o.CompletedItems
	estimatedTimePerItem := avgTime * time.Duration(o.ComplexityFactor)
	estimatedRemainingTime := time.Duration(float64(estimatedTimePerItem) * float64(remainingItems) * o.TrafficFactor * o.WeatherFactor)

	// 考虑高峰时段影响
	if isPeakHour() {
		estimatedRemainingTime = time.Duration(float64(estimatedRemainingTime) * 1.2)
	}

	return estimatedRemainingTime
}

// 判断是否是高峰时段
func isPeakHour() bool {
	now := time.Now()
	hour := now.Hour()
	return (hour >= 11 && hour <= 13) || (hour >= 17 && hour <= 19)
}

// 动态ETA（Estimated Time of Arrival）算法是一种用于预测订单完成或配送到达时间的算法
// 广泛应用在零售和物流领域，特别是在即时配送和电子商务中
func main() {
	order := NewOrder("ORD-001", 10)

	for i := 0; i < 10; i++ {
		time.Sleep(time.Second * 3) // 模拟时间流逝

		// 模拟进度更新
		progress := i + 1
		if rand.Float32() < 0.2 { // 20% 的概率进度不变，模拟延迟
			progress = i
		}
		order.UpdateProgress(progress)

		eta := order.CalculateETA()
		fmt.Printf("Order %s - Progress: %d/%d, ETA: %v, Traffic: %.2f, Weather: %.2f\n",
			order.ID, order.CompletedItems, order.TotalItems, eta.Round(time.Second),
			order.TrafficFactor, order.WeatherFactor)
	}
}
