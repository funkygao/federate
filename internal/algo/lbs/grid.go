package main

import (
	"fmt"
	"math"
)

// Grid 表示地理网格单元，存储在 KV
type Grid struct {
	Code     string
	Level    int
	Center   GeoPoint
	aois     map[string]*AOI
	riders   map[string]*Rider // 改为存储Rider指针
	orderQty int
}

// GridSystem 管理所有网格
type GridSystem struct {
	gridSize  float64
	grids     map[string]*Grid
	fence     FenceService
	aoiLookup map[string]*AOI
}

// NewGridSystem 创建网格系统
func NewGridSystem(gridSize float64) *GridSystem {
	return &GridSystem{
		gridSize:  gridSize,
		grids:     make(map[string]*Grid),
		aoiLookup: make(map[string]*AOI),
	}
}

// LocateGrid 获取坐标对应的网格编码
func (gs *GridSystem) LocateGrid(p GeoPoint) string {
	x := math.Floor(p.Lon / gs.gridSize)
	y := math.Floor(p.Lat / gs.gridSize)
	return fmt.Sprintf("%.0f-%.0f", x, y)
}

// UpdateRiderPosition 更新骑手位置
func (gs *GridSystem) UpdateRiderPosition(rider *Rider, pos GeoPoint) {
	// 移除旧网格
	if oldGrid, exists := gs.grids[rider.GridCode]; exists {
		delete(oldGrid.riders, rider.ID)
	}

	// 计算新网格
	newCode := gs.LocateGrid(pos)
	rider.UpdatePosition(pos, newCode)

	// 添加到新网格
	if _, exists := gs.grids[newCode]; !exists {
		gs.grids[newCode] = &Grid{
			Code:   newCode,
			aois:   make(map[string]*AOI),
			riders: make(map[string]*Rider),
		}
	}
	gs.grids[newCode].riders[rider.ID] = rider
}

// FindAOIsInGrid 查找网格内的AOI列表
func (gs *GridSystem) FindAOIsInGrid(gridCode string) []*AOI {
	grid, exists := gs.grids[gridCode]
	if !exists {
		return nil
	}

	aois := make([]*AOI, 0, len(grid.aois))
	for _, aoi := range grid.aois {
		aois = append(aois, aoi)
	}
	return aois
}

// collectRiders 收集相邻网格的骑手（示例实现）
func (gs *GridSystem) collectRiders(baseGrid string, layers int) map[string]bool {
	riders := make(map[string]bool)
	// 示例实现：仅获取当前网格
	if grid, exists := gs.grids[baseGrid]; exists {
		for rider := range grid.riders {
			riders[rider] = true
		}
	}
	return riders
}

// AssignOrderToRider 订单分配逻辑
func (gs *GridSystem) AssignOrderToRider(order *Order) (string, []string) {
	poi := order.POI
	dest := order.DeliveryPoint
	srcGrid := gs.LocateGrid(poi.Location)
	destGrid := gs.LocateGrid(dest)

	candidates := gs.collectRiders(srcGrid, 3)

	var bestRider string
	minDistance := math.MaxFloat64
	for rider := range candidates {
		// 简单示例：选择最近的骑手
		d := HaversineDistance(poi.Location, dest)
		if d < minDistance {
			minDistance = d
			bestRider = rider
		}
	}

	return bestRider, []string{srcGrid, destGrid}
}
