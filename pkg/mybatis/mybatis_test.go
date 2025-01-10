package mybatis

import (
	"log"
	"reflect"
	"strings"
	"testing"

	"federate/pkg/util"
	"github.com/stretchr/testify/assert"
	"github.com/xwb1989/sqlparser"
)

func TestSQLAnalyzer(t *testing.T) {
	testCases := []struct {
		name        string
		id          string
		expected    string
		skip        bool
		countFields int
	}{
		{
			name:     "Insert statement",
			id:       "singleInsert",
			expected: `INSERT INTO st_stock_stream ( tenant_code, remark, version, deleted ) VALUES ( ?, ?, 0, ? )`,
		},
		{
			name:     "Batch insert statement",
			id:       "batchInsert",
			expected: `INSERT INTO st_stock_stream ( tenant_code, remark, version, deleted ) VALUES /* FOREACH_START */ ( ?,?,0,? ) /* FOREACH_END */`,
		},
		{
			name:     "Select statement with include",
			id:       "selectByUuid",
			expected: `SELECT id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version FROM st_stock_stream WHERE deleted = 0 AND uuid = ? AND warehouse_no = ? LIMIT 1`,
		},
		{
			name:     "Complex select statement with if and foreach",
			id:       "json",
			expected: `SELECT DISTINCT sku, zone_no AS zoneNo FROM st_stock WHERE deleted = 0 AND warehouse_no = ? AND zone_no IN (/* FOREACH_START */ ? /* FOREACH_END */) AND (/* FOREACH_START */ extend_content ->> ? = ? /* FOREACH_END */)`,
		},
		{
			name:     "Complex update if and foreach",
			id:       "updateCheckResult",
			expected: `UPDATE st_device_stock_check_result SET foo=? WHERE /* FOREACH_START */ ( deleted = 0 AND id = ? AND warehouse_no = ? AND business_no = ? AND status = ? ) /* FOREACH_END */`,
		},
		{
			name:     "Complex order by",
			id:       "selectStockForManualLocating",
			expected: `SELECT m.id, m.warehouse_no warehouseNo FROM st_stock m LEFT JOIN st_lot_detail ld ON m.sku = ld.sku AND m.lot_no = ld.lot_no AND ld.deleted = 0 LEFT JOIN st_lot_shelf_life l ON m.tenant_code = l.tenant_code AND m.sku = l.sku AND m.lot_no = l.lot_no WHERE m.deleted = 0 AND l.deleted = 0 ORDER BY ld.expiration_date ASC, ld.expiration_date DESC, l.extend_content -> ? ASC`,
		},
		{
			name:     "queryZoneStock",
			id:       "queryZoneStock",
			expected: `SELECT sku FROM st_stock WHERE deleted = 0 AND warehouse_no = ? AND sku in (/* FOREACH_START */ ? /* FOREACH_END */) AND zone_no in (/* FOREACH_START */ ? /* FOREACH_END */) group by sku, zone_no`,
		},
		{
			name:     "selectLocationStock",
			id:       "selectLocationStock",
			expected: `SELECT COUNT(distinct sku) AS skuQty, zone_no AS zoneNo, SUM(st.frozen_qty) AS frozenQty FROM st_stock st WHERE st.deleted = 0 AND st.warehouse_no = ? GROUP BY st.location_no HAVING statusSum > 0 and statusSum = 0`,
		},
		{
			name:     "refreshPropertiesMap",
			id:       "refreshPropertiesMap",
			skip:     true,
			expected: `UPDATE st_stock_sku set version = version + 1 , extend_content = json_set(extend_content, '', ?) WHERE deleted = 0 AND warehouse_no = ? AND id = ?`,
		},
	}

	xml := NewXMLMapperBuilder("test.xml")
	xml.Prepare()
	stmts, err := xml.Parse()
	assert.NoError(t, err)

	for _, tc := range testCases {
		if !tc.skip {
			t.Run(tc.name, func(t *testing.T) {
				stmtID := "com.goog.wms.StockDao." + tc.id
				stmt := stmts[stmtID]

				assert.Equal(t, tc.expected, strings.TrimSpace(stmt.SQL), tc.id)
			})
		}
	}
}

func TestSplitSQLStatements(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "Multiple semicolons at the end",
			input:    "SELECT * FROM users;;;",
			expected: []string{"SELECT * FROM users"},
		},
		{
			name:     "Simple statements",
			input:    "SELECT * FROM users; INSERT INTO logs VALUES (1, 'test');",
			expected: []string{"SELECT * FROM users", "INSERT INTO logs VALUES (1, 'test')"},
		},
		{
			name:     "Statements with strings containing semicolons",
			input:    "SELECT * FROM users WHERE name = 'John; Doe'; INSERT INTO logs VALUES (1, 'test; with; semicolons');",
			expected: []string{"SELECT * FROM users WHERE name = 'John; Doe'", "INSERT INTO logs VALUES (1, 'test; with; semicolons')"},
		},
		{
			name:     "Statements with comments",
			input:    "SELECT * FROM users; -- This is a comment; with semicolons\nINSERT INTO logs VALUES (1, 'test');",
			expected: []string{"SELECT * FROM users", "-- This is a comment; with semicolons\nINSERT INTO logs VALUES (1, 'test')"},
		},
		{
			name:     "Statements with mixed quotes",
			input:    "SELECT * FROM users WHERE name = 'John''s \"nickname\"'; INSERT INTO logs VALUES (1, \"test\");",
			expected: []string{"SELECT * FROM users WHERE name = 'John''s \"nickname\"'", "INSERT INTO logs VALUES (1, \"test\")"},
		},
		{
			name:     "Statement without semicolon at the end",
			input:    "SELECT * FROM users WHERE id = 1",
			expected: []string{"SELECT * FROM users WHERE id = 1"},
		},
		{
			name:     "Empty input",
			input:    "",
			expected: nil,
		},
		{
			name:     "Multiple semicolons and whitespace",
			input:    ";;  SELECT * FROM users;  ;;  INSERT INTO logs VALUES (1, 'test');;  ;",
			expected: []string{"SELECT * FROM users", "INSERT INTO logs VALUES (1, 'test')"},
		},
		{
			name:     "Semicolons in function calls",
			input:    "SELECT CONCAT('a;', 'b;', 'c;') AS result; INSERT INTO logs VALUES (1, 'test');",
			expected: []string{"SELECT CONCAT('a;', 'b;', 'c;') AS result", "INSERT INTO logs VALUES (1, 'test')"},
		},
	}

	for _, tc := range testCases {
		stmt := Statement{SQL: tc.input}
		t.Run(tc.name, func(t *testing.T) {
			result := stmt.splitSQL()
			assert.Equal(t, tc.expected, result, "Unexpected split result")
		})
	}
}

func TestSplitSQL(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "Multiple semicolons at the end",
			input:    "SET @affected_rows = 0; DELETE  FROM users;;     ;set @affected_rows  = @affected_rows + row_count();  select @affected_rows as rows;",
			expected: []string{"SET @affected_rows = 0", "DELETE  FROM users", "set @affected_rows  = @affected_rows + row_count()", "select @affected_rows as rows"},
		},
	}
	for _, tc := range testCases {
		stmt := Statement{SQL: tc.input}
		t.Run(tc.name, func(t *testing.T) {
			result := stmt.splitSQL()
			assert.Equal(t, tc.expected, result, "Unexpected split result")
		})
	}
}

func TestSQLParser(t *testing.T) {
	sql := `UPDATE st_stock_sku set version = version + 1 , extend_content = json_set(extend_content, '', ?) WHERE deleted = 0 AND warehouse_no = ? AND id = ?`
	sql = `select @affected_rows as rows`
	sql = `SET  @affected_rows = 0`
	stmt, err := sqlparser.Parse(sql)
	assert.NoError(t, err)
	if testing.Verbose() {
		log.Printf("\n%s\nactual type: %+v", util.Beautify(stmt), reflect.TypeOf(stmt))
	}
}

func TestAnalyzeComplexity(t *testing.T) {
	testCases := []struct {
		name    string
		input   string
		score   int
		reasons []string
	}{
		{
			name: "basic",
			input: `
SELECT
  MAX(stock.location_no) AS locationNo,
  capacity.zone_no AS zoneNo,
  capacity.zone_type as zoneType
FROM st_stock stock JOIN st_location_capacity capacity
  ON stock.warehouse_no = capacity.warehouse_no AND stock.location_no = capacity.location_no
WHERE stock.deleted = 0 AND capacity.deleted = 0
  AND stock.warehouse_no = ?
  AND capacity.zone_no IN (?)
  AND capacity.zone_type IN ('cp', 'bp')
GROUP BY stock.location_no, capacity.zone_no,  capacity.zone_type
UNION
SELECT
  max(stock.location_no) AS locationNo,
  capacity.zone_no AS zoneNo,
  capacity.zone_type AS zoneType
FROM st_location_occupy stock JOIN st_location_capacity capacity
  ON stock.warehouse_no = capacity.warehouse_no AND stock.location_no = capacity.location_no
WHERE stock.deleted = 0 AND capacity.deleted = 0
  AND stock.warehouse_no = ?
  AND capacity.zone_no IN (?)
  AND capacity.zone_type in ('cp', 'bp')
GROUP BY stock.location_no, capacity.zone_no, capacity.zone_type
        `,
			score:   11,
			reasons: []string{"GROUP BY", "JOIN", "MAX", "UNION"},
		},
	}
	for _, tc := range testCases {
		stmt := Statement{SQL: tc.input}
		t.Run(tc.name, func(t *testing.T) {
			stmt.Analyze()
			result := stmt.Complexity
			assert.Equal(t, tc.score, result.Score)
			assert.Equal(t, tc.reasons, result.Reasons.SortedValues())
		})
	}
}

func TestMinimalSQL(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "basic",
			input:    `UPDATE      st_check_result SET foo=?   WHERE /* FOREACH_START */ ( deleted = 0 AND warehouse_no = ? AND id = ? ) /* FOREACH_END */`,
			expected: `UPDATE st_check_result SET foo=? WHERE ( deleted = 0 AND warehouse_no = ? AND id = ? )`,
		},
	}
	for _, tc := range testCases {
		stmt := Statement{SQL: tc.input}
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, stmt.MinimalSQL())
		})
	}
}
