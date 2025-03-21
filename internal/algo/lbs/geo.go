package main

import "math"

// GeoPoint 表示地理坐标点
type GeoPoint struct {
	Lon float64 // 经度
	Lat float64 // 纬度
}

// HaversineDistance 计算球面距离（米）
func HaversineDistance(a, b GeoPoint) float64 {
	const R = 6371e3 // 地球半径（米）

	φ1 := a.Lat * math.Pi / 180
	φ2 := b.Lat * math.Pi / 180
	Δφ := (b.Lat - a.Lat) * math.Pi / 180
	Δλ := (b.Lon - a.Lon) * math.Pi / 180

	aCalc := math.Sin(Δφ/2)*math.Sin(Δφ/2) +
		math.Cos(φ1)*math.Cos(φ2)*
			math.Sin(Δλ/2)*math.Sin(Δλ/2)

	c := 2 * math.Atan2(math.Sqrt(aCalc), math.Sqrt(1-aCalc))

	return R * c
}
