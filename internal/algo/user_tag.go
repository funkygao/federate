package main

import (
	"fmt"
)

// UserTag 枚举用户标签
type UserTag int

const (
	Male UserTag = iota
	Female
	Age18To24
	Age25To34
	IncomeHigh
	IncomeMedium
	IncomeLow
	LikeSports
	LikeMusic
	LikeTravel
)

// RoaringBitmap 接口定义
type RoaringBitmap interface {
	Add(x uint32)
	Contains(x uint32) bool
	And(other RoaringBitmap) RoaringBitmap
	AndNot(other RoaringBitmap) RoaringBitmap
	Or(other RoaringBitmap) RoaringBitmap
	ToArray() []uint32
}

// 用户画像管理器
// 每个标签：高16位64K个bucket，每个bucket指针占用空间8B；低16位 BitmapContainer [1024]int64，即64KB
// 每个标签占用空间：64KB + 64KB * 8 = 576KB，1000个标签则 576MB，1万个标签大概5GB就够了
type UserProfiler struct {
	// 每个标签对应一个RBM
	tagBitmaps map[UserTag]RoaringBitmap
}

// 为某个用户打标签
func (up *UserProfiler) TagUser(userId uint32, tags []UserTag) {
	for _, tag := range tags {
		up.tagBitmaps[tag].Add(userId)
	}
}

// 圈人：返回 userId list
func (up *UserProfiler) DemoUserSegement() []uint32 {
	bitmap := up.tagBitmaps[Male].And(up.tagBitmaps[Age18To24]).AndNot(up.tagBitmaps[IncomeHigh])
	return bitmap.ToArray()
}
