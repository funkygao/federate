package concurrent

import (
	"sync/atomic"
)

// Counter is a thread-safe counter
type Counter struct {
	value int64
}

// NewCounter creates a new Counter
func NewCounter() *Counter {
	return &Counter{}
}

// Increment atomically increments the counter
func (c *Counter) Increment() {
	atomic.AddInt64(&c.value, 1)
}

// Value returns the current value of the counter
func (c *Counter) Value() int64 {
	return atomic.LoadInt64(&c.value)
}
