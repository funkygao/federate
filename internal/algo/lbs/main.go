package main

import (
	"fmt"
	"math/rand"
)

func main() {
	gridSystem := NewGridSystem(0.01) // 0.01度≈1km

	cbdAOI := &AOI{
		ID:   "cbd",
		Name: "Central Business District",
		Boundary: []GeoPoint{
			{116.300, 39.900},
			{116.310, 39.900},
			{116.305, 39.910},
		},
		POIs: make(map[string]*POI),
	}

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

	// 模拟骑手位置
	rand.Seed(42)
	for i := 0; i < 5; i++ {
		gridSystem.UpdateRiderPosition(
			fmt.Sprintf("rider_%d", i+1),
			GeoPoint{
				116.300 + rand.Float64()*0.015,
				39.900 + rand.Float64()*0.015,
			},
		)
	}

	// 订单分配演示
	rider, path := gridSystem.AssignOrderToRider(
		restaurant,
		GeoPoint{116.308, 39.907},
	)

	fmt.Println("=== 调度结果 ===")
	fmt.Printf("分配骑手: %s\n", rider)
	fmt.Printf("配送路径网格: %v\n", path)

	// 围栏检测演示
	testPoint := GeoPoint{116.307, 39.901}
	contains := cbdAOI.Contains(testPoint)
	fmt.Printf("\n=== 围栏检测 ===\n坐标 %v\n在CBD区域: %t\n", testPoint, contains)
}
