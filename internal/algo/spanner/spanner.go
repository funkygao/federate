package main

import (
	"time"
)

type Spanner interface {
	BeginTransaction() Transaction
	Read(key string) (interface{}, error)
	Write(key string, value interface{}) error
}

type spanner struct {
	regions  []Region
	truetime TrueTime
	mvcc     MVCC
}

func NewSpanner(regions []Region, truetime TrueTime, mvcc MVCC) Spanner {
	return &spanner{
		regions:  regions,
		truetime: truetime,
		mvcc:     mvcc,
	}
}

func (s *spanner) BeginTransaction() Transaction {
	return NewTransaction("tx-"+time.Now().String(), s.truetime.Now(), s.mvcc)
}

func (s *spanner) Read(key string) (interface{}, error) {
	now := s.truetime.Now()
	return s.mvcc.Read(key, now.Latest)
}

func (s *spanner) Write(key string, value interface{}) error {
	now := s.truetime.Now()
	return s.mvcc.Write(key, value, now.Latest)
}
