package main

import (
	"fmt"
	"time"
)

// OrderStatus 订单状态类型
type OrderStatus int

const (
	Created OrderStatus = iota
	Assigned
	Delivering
	Completed
	Canceled
)

// Order 表示配送订单
type Order struct {
	ID            string
	POI           *POI     // 取货点
	DeliveryPoint GeoPoint // 送货地址
	CreatedAt     time.Time
	AssignedAt    *time.Time
	RiderID       string
	Status        OrderStatus
}

// NewOrder 创建新订单
func NewOrder(poi *POI, dest GeoPoint) *Order {
	return &Order{
		ID:            generateOrderID(),
		POI:           poi,
		DeliveryPoint: dest,
		CreatedAt:     time.Now(),
		Status:        Created,
	}
}

// Assign 分配骑手
func (o *Order) Assign(riderID string) {
	now := time.Now()
	o.RiderID = riderID
	o.AssignedAt = &now
	o.Status = Assigned
}

// 生成订单ID
func generateOrderID() string {
	return fmt.Sprintf("ORD-%d", time.Now().UnixNano())
}
