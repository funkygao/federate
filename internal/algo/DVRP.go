package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Point struct {
	X, Y float64
}

type Order struct {
	ID        int
	Location  Point
	ReadyTime int
}

type Vehicle struct {
	ID       int
	Location Point
	Orders   []Order
}

type DistanceMatrix map[int]map[int]float64

func distance(a, b Point) float64 {
	return math.Sqrt(math.Pow(a.X-b.X, 2) + math.Pow(a.Y-b.Y, 2))
}

func createDistanceMatrix(orders []Order, vehicles []Vehicle) DistanceMatrix {
	matrix := make(DistanceMatrix)
	for i := range orders {
		matrix[orders[i].ID] = make(map[int]float64)
		for j := range orders {
			matrix[orders[i].ID][orders[j].ID] = distance(orders[i].Location, orders[j].Location)
		}
		for j := range vehicles {
			matrix[orders[i].ID][-vehicles[j].ID] = distance(orders[i].Location, vehicles[j].Location)
		}
	}
	return matrix
}

// 一个简单的贪心算法来解决DVRP问题
func assignOrders(orders []Order, vehicles []Vehicle, matrix DistanceMatrix) {
	for _, order := range orders {
		bestVehicle := -1
		minDistance := math.Inf(1)

		for i, vehicle := range vehicles {
			if len(vehicle.Orders) == 0 {
				dist := matrix[order.ID][-vehicle.ID]
				if dist < minDistance {
					minDistance = dist
					bestVehicle = i
				}
			} else {
				lastOrder := vehicle.Orders[len(vehicle.Orders)-1]
				dist := matrix[lastOrder.ID][order.ID]
				if dist < minDistance {
					minDistance = dist
					bestVehicle = i
				}
			}
		}

		if bestVehicle != -1 {
			vehicles[bestVehicle].Orders = append(vehicles[bestVehicle].Orders, order)
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// 创建订单
	orders := make([]Order, 10)
	for i := range orders {
		orders[i] = Order{
			ID:        i,
			Location:  Point{X: rand.Float64() * 100, Y: rand.Float64() * 100},
			ReadyTime: rand.Intn(60),
		}
	}

	// 创建车辆
	vehicles := make([]Vehicle, 3)
	for i := range vehicles {
		vehicles[i] = Vehicle{
			ID:       i + 1,
			Location: Point{X: rand.Float64() * 100, Y: rand.Float64() * 100},
		}
	}

	// 创建距离矩阵
	matrix := createDistanceMatrix(orders, vehicles)

	// 分配订单
	assignOrders(orders, vehicles, matrix)

	// 打印结果
	for _, vehicle := range vehicles {
		fmt.Printf("Vehicle %d:\n", vehicle.ID)
		for _, order := range vehicle.Orders {
			fmt.Printf("  Order %d at (%.2f, %.2f)\n", order.ID, order.Location.X, order.Location.Y)
		}
		fmt.Println()
	}
}
