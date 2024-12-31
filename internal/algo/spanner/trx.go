package main

type Transaction interface {
	Read(key string) (interface{}, error)
	Write(key string, value interface{}) error
	Commit() error
	Rollback() error
}

type transaction struct {
	id        string
	reads     map[string]interface{}
	writes    map[string]interface{}
	timestamp TimeInterval
	mvcc      MVCC
}

func NewTransaction(id string, timestamp TimeInterval, mvcc MVCC) Transaction {
	return &transaction{
		id:        id,
		reads:     make(map[string]interface{}),
		writes:    make(map[string]interface{}),
		timestamp: timestamp,
		mvcc:      mvcc,
	}
}

func (t *transaction) Read(key string) (interface{}, error) {
	if value, ok := t.writes[key]; ok {
		return value, nil
	}
	if value, ok := t.reads[key]; ok {
		return value, nil
	}
	value, err := t.mvcc.Read(key, t.timestamp.Latest)
	if err == nil {
		t.reads[key] = value
	}
	return value, err
}

func (t *transaction) Write(key string, value interface{}) error {
	t.writes[key] = value
	return nil
}

func (t *transaction) Commit() error {
	for key, value := range t.writes {
		if err := t.mvcc.Write(key, value, t.timestamp.Latest); err != nil {
			return err
		}
	}
	return nil
}

func (t *transaction) Rollback() error {
	t.reads = make(map[string]interface{})
	t.writes = make(map[string]interface{})
	return nil
}
