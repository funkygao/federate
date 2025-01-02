package mybatis

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

const testXML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.goog.wms.StockDao">
    <resultMap id="resultMap" type="com.goog.wms.stock.infrastructure.jdbc.main.po.StockStreamPo">
        <id column="id" property="id" jdbcType="BIGINT"/>
        <result column="tenant_code" property="tenantCode" jdbcType="VARCHAR"/>
        <result column="warehouse_no" property="warehouseNo" jdbcType="VARCHAR"/>
        <result column="uuid" property="uuid" jdbcType="VARCHAR"/>
        <result column="business_type" property="businessType" jdbcType="VARCHAR"/>
        <result column="business_no" property="businessNo" jdbcType="VARCHAR"/>
        <result column="remark" property="remark" jdbcType="VARCHAR"/>
        <result column="version" property="version" jdbcType="INTEGER"/>
    </resultMap>

    <sql id="queryColumns">
        <![CDATA[
            id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version
        ]]>
    </sql>
    <sql id="queryColumns2">
        <include refid="queryColumns"/>, ext
    </sql>

    <insert id="singleInsert" parameterType="com.goog.wms.stock.infrastructure.jdbc.main.po.StockStreamPo" useGeneratedKeys="true" keyProperty="id">
        INSERT INTO st_stock_stream (
            tenant_code, remark, version, deleted
        ) VALUES (
            #{tenantCode, jdbcType=VARCHAR}, #{remark, jdbcType=VARCHAR}, 0,
            <choose>
                <when test="deleted != null and deleted > 0">
                    #{deleted, jdbcType=INTEGER}
                </when>
                <otherwise>
                    0
                </otherwise>
            </choose>
        )
    </insert>

    <insert id="batchInsert" parameterType="java.util.List">
        INSERT INTO st_stock_stream (
            tenant_code, remark, version, deleted
        ) VALUES
        <foreach collection="list" item="item" index="index" separator=",">
            (
            #{item.tenantCode, jdbcType=VARCHAR}, #{item.remark, jdbcType=VARCHAR}, 0,
            <choose>
                <when test="item.deleted != null and item.deleted >= 0">
                    #{item.deleted, jdbcType=INTEGER}
                </when>
                <otherwise>
                    0
                </otherwise>
            </choose>
            )
        </foreach>
    </insert>

    <select id="selectByUuid" parameterType="com.goog.wms.stock.domain.main.entity.query.StockStreamQuery" resultMap="resultMap">
        SELECT
        <include refid="queryColumns"/>
        FROM st_stock_stream
        <where>
            deleted = 0
            AND uuid = #{uuid, jdbcType=VARCHAR}
            AND warehouse_no = #{warehouseNo, jdbcType=VARCHAR}
        </where>
        LIMIT 1
    </select>

    <select id="json" parameterType="com.goog.wms.stock.domain.capacity.entity.query.ZoneRecommendQuery" resultType="com.jdwl.wms.stock.infrastructure.jdbc.capacity.po.ZoneSkuPo">
        SELECT
          DISTINCT
          sku,
          zone_no AS zoneNo
        FROM st_stock
        WHERE deleted = 0
          AND warehouse_no = #{warehouseNo, jdbcType=VARCHAR}
          <if test="null != zoneNoList and !zoneNoList.empty">
              AND zone_no IN
              <foreach collection="zoneNoList" item="item" index="index" open="(" separator="," close=")">
                  #{item, jdbcType=VARCHAR}
              </foreach>
          </if>
        AND
        <foreach collection="skuAttributePairs" item="item" open="(" close=")" separator="AND">
            extend_content ->> #{item.jsonKey} = #{item.value}
        </foreach>
    </select>

    <update id="updateCheckResult" parameterType="java.util.List">
        <if test="list != null and !list.empty">
            UPDATE st_device_stock_check_result
            <trim prefix="set" suffixOverrides=",">
                <trim prefix="update_user=case" suffix="end,">
                    <foreach collection="list" item="item" index="index">
                        WHEN
                        <include refid="commonUpdate4CheckFinishedClauseOfItem"/>
                        THEN #{item.updateUser,jdbcType=VARCHAR}
                    </foreach>
                </trim>
                <trim prefix="check_type=case" suffix="end,">
                    <foreach collection="list" item="item" index="index">
                        WHEN
                        <include refid="commonUpdate4CheckFinishedClauseOfItem"/>
                        THEN #{item.checkType,jdbcType=DECIMAL}
                    </foreach>
                </trim>
                <trim prefix="difference_qty=case" suffix="end,">
                    <foreach collection="list" item="item" index="index">
                        WHEN
                        <include refid="commonUpdate4CheckFinishedClauseOfItem"/>
                        THEN #{item.differenceQty,jdbcType=DECIMAL}
                    </foreach>
                </trim>
                update_time = now()
                , status = 20
                , version = version + 1
            </trim>
            WHERE
            <foreach collection="list" item="item" index="index" open="" separator="or" close="">
                (<include refid="commonUpdate4CheckFinishedClauseOfItem"/>)
            </foreach>
        </if>
    </update>

    <sql id="commonUpdate4CheckFinishedClauseOfItem">
        deleted = 0
        AND id = #{item.id, jdbcType=BIGINT}
        AND warehouse_no = #{item.warehouseNo, jdbcType=VARCHAR}
        AND business_no = #{item.businessNo, jdbcType=VARCHAR}
        <if test="null != item.previousStatus">
            AND status = #{item.previousStatus.code, jdbcType=INTEGER}
        </if>
    </sql>

    <select id="selectStockForManualLocating">
        SELECT
        m.id,
        m.warehouse_no warehouseNo
        FROM st_stock m
        LEFT JOIN st_lot_detail ld ON m.sku = ld.sku AND m.lot_no = ld.lot_no AND ld.deleted = 0
        <if test="null != shelfLifeManagement and shelfLifeManagement == true and ((null != sourceModule and sourceModule.code == &apos;wms-outbound-plan&apos;) or null != perShelfLifeDays)">
            LEFT JOIN st_lot_shelf_life l ON m.tenant_code = l.tenant_code AND m.sku = l.sku AND m.lot_no = l.lot_no
        </if>
        <include refid="selectStockForManualLocatingCondition"/>
        <if test="null != expirationOrderDirection and expirationOrderDirection.key == &apos;asc&apos;">
            ORDER BY ld.expiration_date ASC
        </if>
        <if test="null != expirationOrderDirection and expirationOrderDirection.key == &apos;desc&apos;">
            ORDER BY ld.expiration_date DESC
        </if>
    </select>

    <sql id="selectStockForManualLocatingCondition">
        WHERE m.deleted = 0
        <if test="null != shelfLifeManagement and shelfLifeManagement == true">
            <if test="null != perShelfLifeDays">
                AND l.deleted = 0
            </if>
        </if>
    </sql>
</mapper>`

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
			expected: `INSERT INTO st_stock_stream ( tenant_code, remark, version, deleted ) VALUES /* FOREACH_START */ ( ?, ?, 0, ? ) /* FOREACH_END *//* FOREACH_ITEM */`,
		},
		{
			name:     "Select statement with include",
			id:       "selectByUuid",
			expected: `SELECT id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version FROM st_stock_stream WHERE deleted = 0 AND uuid = ? AND warehouse_no = ? LIMIT 1`,
		},
		{
			name:     "Complex select statement with if and foreach",
			id:       "json",
			expected: `SELECT DISTINCT sku, zone_no AS zoneNo FROM st_stock WHERE deleted = 0 AND warehouse_no = ? AND zone_no IN ( ? , ? ) AND ( extend_content ->> ? = ? AND extend_content ->> ? = ? )`,
		},
		{
			name:     "Complex update if and foreach",
			id:       "updateCheckResult",
			expected: `UPDATE st_device_stock_check_result set update_user=case WHEN deleted = 0 AND id = ? AND warehouse_no = ? AND business_no = ? AND status = ? THEN ? end, check_type=case WHEN deleted = 0 AND id = ? AND warehouse_no = ? AND business_no = ? AND status = ? THEN ? end, difference_qty=case WHEN deleted = 0 AND id = ? AND warehouse_no = ? AND business_no = ? AND status = ? THEN ? end, update_time = now() , status = 20 , version = version + 1 WHERE (deleted = 0 AND id = ? AND warehouse_no = ? AND business_no = ? AND status = ?) or (deleted = 0 AND id = ? AND warehouse_no = ? AND business_no = ? AND status = ?)`,
		},
		{
			name:     "Complex order by",
			id:       "selectStockForManualLocating",
			expected: `SELECT m.id, m.warehouse_no warehouseNo FROM st_stock m LEFT JOIN st_lot_detail ld ON m.sku = ld.sku AND m.lot_no = ld.lot_no AND ld.deleted = 0 LEFT JOIN st_lot_shelf_life l ON m.tenant_code = l.tenant_code AND m.sku = l.sku AND m.lot_no = l.lot_no WHERE m.deleted = 0 AND l.deleted = 0 ORDER BY ld.expiration_date ASC, ld.expiration_date DESC`,
		},
	}

	xml := NewXMLMapperBuilder("")
	stmts, err := xml.ParseString(testXML)
	assert.NoError(t, err)

	assert.Equal(t, "id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version", xml.SqlFragments["queryColumns"])
	// recursive ref
	assert.Equal(t, "id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version, ext", xml.SqlFragments["queryColumns2"])
	assert.True(t, len(xml.SqlFragments["commonUpdate4CheckFinishedClauseOfItem"]) > 10)

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stmtID := "com.goog.wms.StockDao." + tc.id
			stmt := stmts[stmtID]

			assert.Equal(t, tc.expected, strings.TrimSpace(stmt.SQL), tc.id)
		})
	}
}
