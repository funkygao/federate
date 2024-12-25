package main

import (
	"container/heap"
	"sync"
	"time"
)

// DelayQueue 实现了一个延迟消息队列：最小堆
type DelayQueue struct {
	items *delayHeap
	mu    sync.Mutex
}

type delayItem struct {
	msg       Message
	readyTime time.Time
	index     int
}

type delayHeap []*delayItem

func (h delayHeap) Len() int           { return len(h) }
func (h delayHeap) Less(i, j int) bool { return h[i].readyTime.Before(h[j].readyTime) }
func (h delayHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *delayHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*delayItem)
	item.index = n
	*h = append(*h, item)
}

func (h *delayHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*h = old[0 : n-1]
	return item
}

func NewDelayQueue() *DelayQueue {
	return &DelayQueue{
		items: &delayHeap{},
	}
}

func (dq *DelayQueue) Add(msg Message) {
	dq.mu.Lock()
	defer dq.mu.Unlock()
	item := &delayItem{
		msg:       msg,
		readyTime: msg.ReadyTime(),
	}
	heap.Push(dq.items, item)
}

func (dq *DelayQueue) Poll() (Message, bool) {
	dq.mu.Lock()
	defer dq.mu.Unlock()
	if dq.items.Len() == 0 {
		return Message{}, false
	}
	item := heap.Pop(dq.items).(*delayItem)
	if time.Now().Before(item.readyTime) {
		heap.Push(dq.items, item)
		return Message{}, false
	}
	return item.msg, true
}
