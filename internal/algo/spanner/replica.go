package main

import (
	"time"
)

type Replica interface {
	ID() string
	ApplyWrite(key string, value interface{}, timestamp time.Time) error
	Read(key string, timestamp time.Time) (interface{}, error)
}

type replica struct {
	id   string
	mvcc MVCC
}

func NewReplica(id string, mvcc MVCC) Replica {
	return &replica{id: id, mvcc: mvcc}
}

func (r *replica) ID() string {
	return r.id
}

func (r *replica) ApplyWrite(key string, value interface{}, timestamp time.Time) error {
	return r.mvcc.Write(key, value, timestamp)
}

func (r *replica) Read(key string, timestamp time.Time) (interface{}, error) {
	return r.mvcc.Read(key, timestamp)
}
