package main

import (
	"fmt"
	"time"
)

// Product 表示一个产品
type Product struct {
	ID   string
	Name string
}

// InventoryLocation 表示一个库存位置
type InventoryLocation struct {
	ID       string
	Name     string
	Quantity map[string]int // 每个产品的库存数量
}

type Order struct {
	ID        string
	ProductID string
	Quantity  int
	DueDate   time.Time
	Location  string // 订单关联的库存位置
}

// ParallelInventorySystem 管理平行库存系统
type ParallelInventorySystem struct {
	Products  map[string]Product
	Locations map[string]*InventoryLocation
	Orders    []Order
}

func NewParallelInventorySystem() *ParallelInventorySystem {
	return &ParallelInventorySystem{
		Products:  make(map[string]Product),
		Locations: make(map[string]*InventoryLocation),
		Orders:    []Order{},
	}
}

func (pis *ParallelInventorySystem) AddProduct(p Product) {
	pis.Products[p.ID] = p
}

func (pis *ParallelInventorySystem) AddLocation(loc InventoryLocation) {
	pis.Locations[loc.ID] = &loc
}

func (pis *ParallelInventorySystem) AddInventory(locationID, productID string, quantity int) {
	if loc, exists := pis.Locations[locationID]; exists {
		if loc.Quantity == nil {
			loc.Quantity = make(map[string]int)
		}
		loc.Quantity[productID] += quantity
	}
}

func (pis *ParallelInventorySystem) AddOrder(o Order) {
	pis.Orders = append(pis.Orders, o)
}

// CalculateATP 计算特定产品在给定日期的 ATP(Available to Promise)
func (pis *ParallelInventorySystem) CalculateATP(productID string, date time.Time) map[string]int {
	atp := make(map[string]int)

	// 计算每个位置的 ATP
	for locID, loc := range pis.Locations {
		quantity, exists := loc.Quantity[productID]
		if !exists {
			continue
		}

		availableQuantity := quantity

		// 减去该位置在指定日期前的所有订单数量
		for _, order := range pis.Orders {
			if order.ProductID == productID && order.Location == locID && order.DueDate.Before(date) {
				availableQuantity -= order.Quantity
			}
		}

		if availableQuantity > 0 {
			atp[locID] = availableQuantity
		}
	}

	return atp
}

func main() {
	pis := NewParallelInventorySystem()

	pis.AddProduct(Product{ID: "P1", Name: "Product 1"})

	pis.AddLocation(InventoryLocation{ID: "L1", Name: "Warehouse A"})
	pis.AddLocation(InventoryLocation{ID: "L2", Name: "Warehouse B"})

	pis.AddInventory("L1", "P1", 100)
	pis.AddInventory("L2", "P1", 150)

	pis.AddOrder(Order{ID: "O1", ProductID: "P1", Quantity: 30, DueDate: time.Now().Add(24 * time.Hour), Location: "L1"})
	pis.AddOrder(Order{ID: "O2", ProductID: "P1", Quantity: 50, DueDate: time.Now().Add(48 * time.Hour), Location: "L2"})

	// 计算 ATP
	atp := pis.CalculateATP("P1", time.Now().Add(72*time.Hour))

	fmt.Println("ATP for Product P1 in 3 days:")
	for locID, quantity := range atp {
		fmt.Printf("Location %s: %d\n", locID, quantity)
	}

	// 计算总 ATP
	totalATP := 0
	for _, quantity := range atp {
		totalATP += quantity
	}
	fmt.Printf("Total ATP: %d\n", totalATP)
}
