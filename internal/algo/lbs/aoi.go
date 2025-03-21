package main

import "sync"

// AOI 表示兴趣区域（地理围栏）
type AOI struct {
	ID        string
	Name      string
	Boundary  []GeoPoint // 围栏边界
	POIs      map[string]*POI
	gridCodes map[string]bool
	lock      sync.RWMutex
	fence     FenceService
}

// Contains 判断点是否在AOI范围内
func (a *AOI) Contains(point GeoPoint) bool {
	return a.fence.Contains(point, a.Boundary)
}

// LinkPOI 关联POI到当前AOI
func (a *AOI) LinkPOI(poi *POI) bool {
	if !a.Contains(poi.Location) {
		return false
	}

	a.lock.Lock()
	defer a.lock.Unlock()
	a.POIs[poi.ID] = poi
	return true
}

// GetGridCodes 获取关联的网格编码列表
func (a *AOI) GetGridCodes() []string {
	a.lock.RLock()
	defer a.lock.RUnlock()
	codes := make([]string, 0, len(a.gridCodes))
	for code := range a.gridCodes {
		codes = append(codes, code)
	}
	return codes
}
