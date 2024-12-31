package main

import (
	"math/rand"
	"time"
)

// TrueTime represents Google's TrueTime API
type TrueTime interface {
	Now() TimeInterval
}

// TimeInterval represents a time interval with earliest and latest possible times
type TimeInterval struct {
	Earliest time.Time
	Latest   time.Time
}

type truetime struct {
	maxUncertainty time.Duration
}

// NewTrueTime creates a new TrueTime instance with the specified maximum uncertainty
func NewTrueTime(maxUncertainty time.Duration) TrueTime {
	return &truetime{maxUncertainty: maxUncertainty}
}

// Now returns a TimeInterval representing the current time with uncertainty
func (tt *truetime) Now() TimeInterval {
	now := time.Now()
	uncertainty := time.Duration(rand.Int63n(int64(tt.maxUncertainty)))
	return TimeInterval{
		Earliest: now.Add(-uncertainty / 2),
		Latest:   now.Add(uncertainty / 2),
	}
}

// After checks if the given time is definitely after the current time
func After(interval TimeInterval, t time.Time) bool {
	return interval.Earliest.After(t)
}

// Before checks if the given time is definitely before the current time
func Before(interval TimeInterval, t time.Time) bool {
	return interval.Latest.Before(t)
}
