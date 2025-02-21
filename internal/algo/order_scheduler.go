package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/robfig/cron/v3"
)

const (
	timeWheelSlots = 60 // 时间轮的槽数，假设每秒一个槽
)

var (
	rdb *redis.Client = redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	ctx               = context.Background()
)

// 时间轮调度器
type TimeWheelScheduler struct {
	cron      *cron.Cron
	slots     [timeWheelSlots]map[string]bool // 时间格，存储哨兵节点
	slotMutex sync.Mutex
	ip        string // 当前服务器的 IP
}

func NewTimeWheelScheduler(ip string) *TimeWheelScheduler {
	return &TimeWheelScheduler{
		cron:  cron.New(),
		ip:    ip,
		slots: [timeWheelSlots]map[string]bool{},
	}
}

// 添加任务到时间轮
func (t *TimeWheelScheduler) AddTask(dispatchTime time.Time, orderID string) error {
	// 计算时间格索引
	slotIndex := int(dispatchTime.Second()) % timeWheelSlots

	// 在 Redis 中存储订单数据
	key := fmt.Sprintf("%s_%s", dispatchTime.Format("2006-01-02 15:04:05"), t.ip)
	err := rdb.Set(ctx, key, orderID, 0).Err()
	if err != nil {
		return err
	}

	// 在时间格中设置哨兵节点
	t.slotMutex.Lock()
	defer t.slotMutex.Unlock()
	if t.slots[slotIndex] == nil {
		t.slots[slotIndex] = make(map[string]bool)
	}
	t.slots[slotIndex][key] = true

	return nil
}

// 启动时间轮
func (t *TimeWheelScheduler) Start() {
	// 每秒触发一次时间轮推进
	t.cron.AddFunc("@every 1s", t.advance)
	t.cron.Start()
}

// 时间轮推进
func (t *TimeWheelScheduler) advance() {
	currentTime := time.Now()
	slotIndex := int(currentTime.Second()) % timeWheelSlots

	t.slotMutex.Lock()
	defer t.slotMutex.Unlock()

	// 处理当前时间格中的任务
	for key := range t.slots[slotIndex] {
		// 从 Redis 中获取订单数据
		orderID, err := rdb.Get(ctx, key).Result()
		if err != nil {
			log.Printf("Failed to get order from Redis: %v", err)
			continue
		}

		// 异步处理订单
		go t.processOrder(orderID)

		// 删除哨兵节点
		delete(t.slots[slotIndex], key)
	}
}

// 处理订单
func (t *TimeWheelScheduler) processOrder(orderID string) {
	// 模拟：该订单有问题，5秒钟后再处理
	log.Printf("Processing order: %s failed, rescheduled for 5s later", orderID)
	t.AddTask(time.Now().Add(5*time.Second), orderID)
}

func main() {
	// 创建时间轮调度器
	scheduler := NewTimeWheelScheduler("192.168.1.1")
	scheduler.Start()

	// 模拟添加订单任务
	dispatchTime := time.Now().Add(7 * time.Second)
	log.Printf("Will process order at: %v", dispatchTime)
	err := scheduler.AddTask(dispatchTime, "order123")
	if err != nil {
		log.Fatalf("Failed to add task: %v", err)
	}

	// 保持程序运行
	select {}
}
