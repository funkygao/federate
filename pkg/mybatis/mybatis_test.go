package mybatis

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/xwb1989/sqlparser"
)

func TestSQLAnalyzer(t *testing.T) {
	testCases := []struct {
		name        string
		id          string
		expected    string
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
			expected: `UPDATE st_stock_sku set version = version + 1 , extend_content = json_set(extend_content, '', ?) WHERE deleted = 0 AND warehouse_no = ? AND id = ?`,
		},
	}

	xml := NewXMLMapperBuilder("test.xml")
	xml.Prepare()
	stmts, err := xml.Parse()
	assert.NoError(t, err)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stmtID := "com.goog.wms.StockDao." + tc.id
			stmt := stmts[stmtID]

			assert.Equal(t, tc.expected, strings.TrimSpace(stmt.SQL), tc.id)
		})
	}
}

func TestSQLParser(t *testing.T) {
	sql := `UPDATE st_stock_sku set version = version + 1 , extend_content = json_set(extend_content, '', ?) WHERE deleted = 0 AND warehouse_no = ? AND id = ?`
	_, err := sqlparser.Parse(sql)
	assert.NoError(t, err)
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

	analyzer := NewSQLAnalyzer(nil, nil)
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := analyzer.splitSQLStatements(tc.input)
			assert.Equal(t, tc.expected, result, "Unexpected split result")
		})
	}
}
