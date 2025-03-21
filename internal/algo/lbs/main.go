package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	// 初始化网格系统（0.01度≈1km）
	gridSystem := NewGridSystem(0.01)

	// 创建测试AOI和POI
	restaurant := createTestData(gridSystem)

	// 初始化骑手系统
	initRiders(gridSystem, 50)

	// 创建测试订单
	order := NewOrder(restaurant, GeoPoint{116.308, 39.907})

	// 执行订单分配（使用正确的参数）
	riderID, path := gridSystem.AssignOrderToRider(order)
	if riderID != "" {
		order.Assign(riderID)
		fmt.Printf("=== 订单分配成功 ===\n订单ID: %s\n分配骑手: %s\n路径: %v\n",
			order.ID, riderID, path)
	} else {
		fmt.Println("没有可用骑手")
	}
}

// 初始化测试数据
func createTestData(gs *GridSystem) *POI {
	// 创建AOI
	cbdAOI := NewAOI("cbd", "Central Business District", []GeoPoint{
		{116.300, 39.900},
		{116.310, 39.900},
		{116.305, 39.910},
	})

	// 创建POI
	restaurant := &POI{
		ID:           "restaurant_001",
		Name:         "Premium Steakhouse",
		Location:     GeoPoint{116.305, 39.905},
		ServiceRange: 800,
	}

	// 关联POI到AOI
	if cbdAOI.Contains(restaurant.Location) {
		cbdAOI.LinkPOI(restaurant)
	}
	return restaurant
}

// 初始化骑手
func initRiders(gs *GridSystem, count int) {
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < count; i++ {
		rider := NewRider(fmt.Sprintf("rider_%d", i+1))
		rider.SetStatus("online")

		// 随机位置
		pos := GeoPoint{
			116.300 + rand.Float64()*0.015,
			39.900 + rand.Float64()*0.015,
		}
		gs.UpdateRiderPosition(rider, pos)
	}
}
