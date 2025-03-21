package main

import (
	"fmt"
	"math"
)

// Grid 表示地理网格单元
type Grid struct {
	Code     string   // 网格编码
	Level    int      // 层级
	Center   GeoPoint // 中心点
	aois     map[string]*AOI
	riders   map[string]bool // 骑手集合
	orderQty int             // 订单数量
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
func (gs *GridSystem) UpdateRiderPosition(riderID string, pos GeoPoint) {
	oldGrids := make(map[string]bool)
	// 移除旧位置
	for _, grid := range gs.grids {
		if _, exists := grid.riders[riderID]; exists {
			delete(grid.riders, riderID)
			oldGrids[grid.Code] = true
		}
	}

	// 添加新位置
	newCode := gs.LocateGrid(pos)
	if _, exists := gs.grids[newCode]; !exists {
		gs.grids[newCode] = &Grid{
			Code:   newCode,
			aois:   make(map[string]*AOI),
			riders: make(map[string]bool),
		}
	}
	gs.grids[newCode].riders[riderID] = true
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
func (gs *GridSystem) AssignOrderToRider(poi *POI, dest GeoPoint) (string, []string) {
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
