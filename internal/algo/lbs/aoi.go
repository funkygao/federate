package main

// AOI 表示兴趣区域（地理围栏），存储在 PostGIS，建立R-Tree索引
// AOI热力分布，存储在 Apache Ignite
type AOI struct {
	ID       string
	Name     string
	Boundary []GeoPoint      // 围栏边界
	POIs     map[string]*POI // 里面包含了哪些 POI

	fence FenceService
}

// AOI边界存储：PostGIS
// CREATE TABLE aoi_boundaries (
//	id VARCHAR(36) PRIMARY KEY,
//	name VARCHAR(255),
//	boundary GEOMETRY(Polygon, 4326) -- WGS84坐标系
// );
//
// CREATE INDEX aoi_boundary_gist ON aoi_boundaries USING GIST (boundary);
func NewAOI(id, name string, boundary []GeoPoint) *AOI {
	return &AOI{
		ID:       id,
		Name:     name,
		Boundary: boundary,
		POIs:     make(map[string]*POI),
		fence:    FenceService{},
	}
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

	a.POIs[poi.ID] = poi
	return true
}
