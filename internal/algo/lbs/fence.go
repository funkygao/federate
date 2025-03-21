package main

import "math"

// FenceService 提供地理围栏相关算法
type FenceService struct{}

// Contains 判断点是否在多边形内（射线法实现）
func (fs *FenceService) Contains(point GeoPoint, polygon []GeoPoint) bool {
	if !fs.inBoundingBox(point, polygon) {
		return false
	}

	inside := false
	for i, j := 0, len(polygon)-1; i < len(polygon); i++ {
		if ((polygon[i].Lat > point.Lat) != (polygon[j].Lat > point.Lat)) &&
			(point.Lon < (polygon[j].Lon-polygon[i].Lon)*(point.Lat-polygon[i].Lat)/
				(polygon[j].Lat-polygon[i].Lat)+polygon[i].Lon) {
			inside = !inside
		}
		j = i
	}
	return inside
}

// inBoundingBox 快速外包矩形检测
func (fs *FenceService) inBoundingBox(p GeoPoint, polygon []GeoPoint) bool {
	minLon, maxLon := math.MaxFloat64, -math.MaxFloat64
	minLat, maxLat := math.MaxFloat64, -math.MaxFloat64

	for _, point := range polygon {
		minLon = math.Min(minLon, point.Lon)
		maxLon = math.Max(maxLon, point.Lon)
		minLat = math.Min(minLat, point.Lat)
		maxLat = math.Max(maxLat, point.Lat)
	}

	return p.Lon >= minLon && p.Lon <= maxLon &&
		p.Lat >= minLat && p.Lat <= maxLat
}
