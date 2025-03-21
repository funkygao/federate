package main

import (
	"time"
)

// Rider 表示配送骑手，实时位置存在 RedisGEO
type Rider struct {
	ID        string
	Status    string    // 状态（online/busy/offline）
	Location  GeoPoint  // 最新位置
	GridCode  string    // 当前所在网格
	UpdatedAt time.Time // 最后更新时间
}

func NewRider(id string) *Rider {
	return &Rider{
		ID:     id,
		Status: "offline",
	}
}

// UpdatePosition 线程安全的位置更新
func (r *Rider) UpdatePosition(pos GeoPoint, gridCode string) {
	r.Location = pos
	r.GridCode = gridCode
	r.UpdatedAt = time.Now()
}

// SetStatus 状态变更方法
func (r *Rider) SetStatus(status string) {
	r.Status = status
}
