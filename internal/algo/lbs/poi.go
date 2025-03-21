package main

// POI 表示兴趣点实体（如餐厅），存储在 PostGIS
type POI struct {
	ID           string   // 唯一标识
	Name         string   // 名称
	Location     GeoPoint // 坐标位置
	ServiceRange float64  // 服务半径（米）
}

// InServiceRange 判断坐标是否在POI服务范围内
func (p *POI) InServiceRange(point GeoPoint) bool {
	return HaversineDistance(p.Location, point) <= p.ServiceRange
}
