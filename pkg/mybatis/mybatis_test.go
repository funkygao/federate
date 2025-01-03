package mybatis

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
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
			expected: `SELECT COUNT(distinct sku) AS skuQty, zone_no AS zoneNo, SUM(st.frozen_qty) AS frozenQty FROM st_stock st WHERE st.deleted = 0 AND st.warehouse_no = ? GROUP BY st.location_no HAVING statusSum > 0 AND statusSum = 0`,
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
