package main

import (
	"fmt"
	"time"
)

type Product struct {
	ID       string
	Name     string
	Quantity int
}

type Order struct {
	ID        string
	ProductID string
	Quantity  int
	DueDate   time.Time
}

type Inventory struct {
	Products map[string]*Product
}

// ATPSystem manages the ATP(Available to Promise) calculation process.
type ATPSystem struct {
	Inventory *Inventory
	Orders    []Order
}

func NewATPSystem() *ATPSystem {
	return &ATPSystem{
		Inventory: &Inventory{
			Products: make(map[string]*Product),
		},
		Orders: []Order{},
	}
}

func (a *ATPSystem) AddProduct(p Product) {
	a.Inventory.Products[p.ID] = &p
}

func (a *ATPSystem) AddOrder(o Order) {
	a.Orders = append(a.Orders, o)
}

// CalculateATP calculates the Available to Promise quantity for a given product at a specific date.
func (a *ATPSystem) CalculateATP(productID string, date time.Time) int {
	product, exists := a.Inventory.Products[productID]
	if !exists {
		return 0
	}

	availableQuantity := product.Quantity

	for _, order := range a.Orders {
		if order.ProductID == productID && order.DueDate.Before(date) {
			availableQuantity -= order.Quantity
		}
	}

	if availableQuantity < 0 {
		return 0
	}

	return availableQuantity
}

func main() {
	atpSystem := NewATPSystem()

	atpSystem.AddProduct(Product{ID: "P1", Name: "Product 1", Quantity: 100})
	atpSystem.AddProduct(Product{ID: "P2", Name: "Product 2", Quantity: 150})

	atpSystem.AddOrder(Order{ID: "O1", ProductID: "P1", Quantity: 30, DueDate: time.Now().Add(24 * time.Hour)})
	atpSystem.AddOrder(Order{ID: "O2", ProductID: "P1", Quantity: 20, DueDate: time.Now().Add(48 * time.Hour)})

	atp := atpSystem.CalculateATP("P1", time.Now().Add(72*time.Hour))
	fmt.Printf("ATP for Product 1 in 3 days: %d\n", atp)

	atp = atpSystem.CalculateATP("P2", time.Now().Add(72*time.Hour))
	fmt.Printf("ATP for Product 2 in 3 days: %d\n", atp)
}
