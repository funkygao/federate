package mybatis

import (
	"database/sql"
	"fmt"

	_ "github.com/mattn/go-sqlite3"
)

type DB struct {
	*sql.DB

	enabled bool
}

func NewDB(dbPath string) (*DB, error) {
	if dbPath == "" {
		return &DB{nil, false}, nil
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("error opening database: %v", err)
	}

	return &DB{db, true}, nil
}

func (db *DB) InitTables() error {
	if !db.enabled {
		return nil
	}

	tables := []string{
		`CREATE TABLE IF NOT EXISTS Statements (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			filename TEXT,
			statement_id TEXT,
			statement_type TEXT,
			raw_sql TEXT,
			processed_sql TEXT,
			timeout INTEGER
		)`,
		`CREATE TABLE IF NOT EXISTS Tables (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			statement_id INTEGER,
			table_name TEXT,
			FOREIGN KEY (statement_id) REFERENCES Statements(id)
		)`,
		`CREATE TABLE IF NOT EXISTS Fields (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			statement_id INTEGER,
			field_name TEXT,
			FOREIGN KEY (statement_id) REFERENCES Statements(id)
		)`,
		`CREATE TABLE IF NOT EXISTS Joins (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			statement_id INTEGER,
			join_type TEXT,
			join_condition TEXT,
			FOREIGN KEY (statement_id) REFERENCES Statements(id)
		)`,
		`CREATE TABLE IF NOT EXISTS Complexity (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			statement_id INTEGER,
			has_subquery BOOLEAN,
			has_union BOOLEAN,
			has_distinct BOOLEAN,
			has_order_by BOOLEAN,
			has_limit BOOLEAN,
			has_offset BOOLEAN,
			FOREIGN KEY (statement_id) REFERENCES Statements(id)
		)`,
		`CREATE TABLE IF NOT EXISTS AggregationFunctions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			statement_id INTEGER,
			function_name TEXT,
			FOREIGN KEY (statement_id) REFERENCES Statements(id)
		)`,
	}

	for _, table := range tables {
		_, err := db.Exec(table)
		if err != nil {
			return fmt.Errorf("error creating table: %v", err)
		}
	}

	db.clearAllTables()

	return nil
}

func (db *DB) clearAllTables() error {
	tables := []string{
		"Statements",
		"Tables",
		"Fields",
		"Joins",
		"Complexity",
		"AggregationFunctions",
	}

	for _, table := range tables {
		_, err := db.Exec(fmt.Sprintf("DELETE FROM %s", table))
		if err != nil {
			return fmt.Errorf("error clearing table %s: %v", table, err)
		}
	}

	// 重置自增ID
	_, err := db.Exec("DELETE FROM sqlite_sequence")
	if err != nil {
		return fmt.Errorf("error resetting auto-increment: %v", err)
	}

	return nil
}

func (db *DB) InsertStatement(stmt *Statement) (int64, error) {
	if !db.enabled {
		return 0, nil
	}

	result, err := db.Exec(`
		INSERT INTO Statements (filename, statement_id, statement_type, raw_sql, processed_sql, timeout)
		VALUES (?, ?, ?, ?, ?, ?)
	`, stmt.Filename, stmt.ID, stmt.Tag, stmt.XMLText, stmt.SQL, stmt.Timeout)
	if err != nil {
		return 0, fmt.Errorf("error inserting statement: %v", err)
	}

	return result.LastInsertId()
}

func (db *DB) InsertTable(stmtID int64, tableName string) error {
	if !db.enabled {
		return nil
	}

	_, err := db.Exec("INSERT INTO Tables (statement_id, table_name) VALUES (?, ?)", stmtID, tableName)
	if err != nil {
		return fmt.Errorf("error inserting table: %v", err)
	}
	return nil
}

func (db *DB) InsertField(stmtID int64, fieldName string) error {
	if !db.enabled {
		return nil
	}

	_, err := db.Exec("INSERT INTO Fields (statement_id, field_name) VALUES (?, ?)", stmtID, fieldName)
	if err != nil {
		return fmt.Errorf("error inserting field: %v", err)
	}
	return nil
}

func (db *DB) InsertJoin(stmtID int64, joinType, joinCondition string) error {
	if !db.enabled {
		return nil
	}

	_, err := db.Exec("INSERT INTO Joins (statement_id, join_type, join_condition) VALUES (?, ?, ?)", stmtID, joinType, joinCondition)
	if err != nil {
		return fmt.Errorf("error inserting join: %v", err)
	}
	return nil
}

func (db *DB) InsertComplexity(stmtID int64, hasSubquery, hasUnion, hasDistinct, hasOrderBy, hasLimit, hasOffset bool) error {
	if !db.enabled {
		return nil
	}

	_, err := db.Exec(`
		INSERT INTO Complexity (statement_id, has_subquery, has_union, has_distinct, has_order_by, has_limit, has_offset)
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`, stmtID, hasSubquery, hasUnion, hasDistinct, hasOrderBy, hasLimit, hasOffset)
	if err != nil {
		return fmt.Errorf("error inserting complexity: %v", err)
	}
	return nil
}

func (db *DB) InsertAggregationFunction(stmtID int64, functionName string) error {
	if !db.enabled {
		return nil
	}

	_, err := db.Exec("INSERT INTO AggregationFunctions (statement_id, function_name) VALUES (?, ?)", stmtID, functionName)
	if err != nil {
		return fmt.Errorf("error inserting aggregation function: %v", err)
	}
	return nil
}
