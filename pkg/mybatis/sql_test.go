package mybatis

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

const testXML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.goog.wms.stock.infrastructure.jdbc.main.dao.StockStreamDao">
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

    <select id="foo" parameterType="com.goog.wms.stock.domain.capacity.entity.query.ZoneRecommendQuery" resultType="com.jdwl.wms.stock.infrastructure.jdbc.capacity.po.ZoneSkuPo">
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
</mapper>`

func TestSQLAnalyzer(t *testing.T) {
	xml := NewXMLAnalyzer()
	err := xml.AnalyzeString(testXML)
	assert.NoError(t, err)

	root := xml.GetRoot()
	assert.NotNil(t, root)

	processor := NewMyBatisProcessor()
	processor.ExtractSQLFragments(root)

	testCases := []struct {
		name     string
		id       string
		expected string
	}{
		{
			name:     "Insert statement",
			id:       "singleInsert",
			expected: `INSERT INTO st_stock_stream ( tenant_code, remark, version, deleted ) VALUES ( ?, ?, 0, ? )`,
		},
		{
			name:     "Batch insert statement",
			id:       "batchInsert",
			expected: `INSERT INTO st_stock_stream ( tenant_code, remark, version, deleted ) VALUES (?, ?, ?, ?), (?, ?, ?, ?)`,
		},
		{
			name:     "Select statement with include",
			id:       "selectByUuid",
			expected: `SELECT id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version FROM st_stock_stream WHERE deleted = 0 AND uuid = ? AND warehouse_no = ? LIMIT 1`,
		},
		{
			name:     "Complex select statement with if and foreach",
			id:       "foo",
			expected: `SELECT DISTINCT sku, zone_no AS zoneNo FROM st_stock WHERE deleted = 0 AND warehouse_no = ? AND zone_no IN (?) AND ( extend_content ->> ? = ? )`,
		},
	}

	assert.Equal(t, "id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version", processor.fragments["queryColumns"])

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			stmt := root.FindElement(".//*[@id='" + tc.id + "']")
			assert.NotNil(t, stmt)

			rawSQL, preprocessedSQL, _ := processor.PreprocessStmt(stmt)
			t.Logf("Original SQL:\n%s\nPreprocessed SQL:\n%s", rawSQL, preprocessedSQL)
			assert.Equal(t, tc.expected, strings.TrimSpace(preprocessedSQL))
		})
	}
}
