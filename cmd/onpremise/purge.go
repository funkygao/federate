package onpremise

import (
	"database/sql"
	"fmt"
	"log"
	"math/rand"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

type TaskPurger struct {
	db        *sql.DB
	batchSize int
}

func doPurgeDb() {
	purger, err := NewTaskPurger(mysqlUser, mysqlPassword, mysqlHost, mysqlPort, purgeBatchSize)
	if err != nil {
		log.Fatalf("%v", err)
	}
	defer purger.Close()

	purger.PurgeAsyncTasks()
}

func NewTaskPurger(user, password, host, port string, batchSize int) (*TaskPurger, error) {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%s)/", user, password, host, port)
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MySQL: %v", err)
	}
	return &TaskPurger{db: db, batchSize: batchSize}, nil
}

func (tp *TaskPurger) Close() {
	tp.db.Close()
}

func (tp *TaskPurger) PurgeAsyncTasks() {
	if err := tp.purgeOldReports(); err != nil {
		log.Printf("Failed to purge old reports: %v", err)
	}

	databases, err := tp.findDatabasesWithTaskTable()
	if err != nil {
		log.Fatalf("Failed to find databases with wms_async_task table: %v", err)
	}

	for _, dbName := range databases {
		if err := tp.purgeOldTasks(dbName); err != nil {
			log.Printf("Failed to purge tasks for database %s: %v", dbName, err)
		} else {
			log.Printf("Successfully purged tasks for database %s", dbName)
		}
	}
}

func (tp *TaskPurger) findDatabasesWithTaskTable() ([]string, error) {
	rows, err := tp.db.Query(`
		SELECT DISTINCT table_schema
		FROM information_schema.tables
		WHERE table_name = 'wms_async_task';
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var databases []string
	for rows.Next() {
		var dbName string
		if err := rows.Scan(&dbName); err != nil {
			return nil, err
		}
		databases = append(databases, dbName)
	}
	return databases, nil
}

func (tp *TaskPurger) purgeOldTasks(dbName string) error {
	for {
		taskIDs, err := tp.getOldTaskIDs(dbName)
		if err != nil {
			return err
		}

		if len(taskIDs) == 0 {
			break
		}

		if err := tp.deleteTasks(dbName, taskIDs); err != nil {
			return err
		}

		tp.sleep()
	}

	return nil
}

func (tp *TaskPurger) getOldTaskIDs(dbName string) ([]int64, error) {
	query := fmt.Sprintf(`
		SELECT id
		FROM %s.wms_async_task
		WHERE status = 4
		AND update_time < DATE_SUB(NOW(), INTERVAL 3 DAY)
		LIMIT %d;
	`, dbName, tp.batchSize)
	rows, err := tp.db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var taskIDs []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		taskIDs = append(taskIDs, id)
	}
	return taskIDs, nil
}

func (tp *TaskPurger) deleteTasks(dbName string, taskIDs []int64) error {
	deleteQuery := fmt.Sprintf(`DELETE FROM %s.wms_async_task WHERE id IN (%s);`, dbName, joinInt64Slice(taskIDs, ","))
	log.Printf("%s", deleteQuery)

	_, err := tp.db.Exec(deleteQuery)
	return err
}

func (tp *TaskPurger) sleep() {
	sleepTime := time.Duration(rand.Intn(10)+5) * time.Millisecond
	time.Sleep(sleepTime)
}

func (tp *TaskPurger) purgeOldReports() error {
	deleteQuery := `DELETE FROM wms_report.rp_event_log WHERE create_time < DATE_SUB(CURDATE(), INTERVAL 3 DAY);`
	log.Printf("%s", deleteQuery)
	_, err := tp.db.Exec(deleteQuery)
	return err
}
