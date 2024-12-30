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
</mapper>`

func TestSQLAnalyzer(t *testing.T) {
	xml := NewXMLAnalyzer()
	err := xml.ReadFromString(testXML)
	assert.NoError(t, err)

	xml.Analyze()
	root := xml.GetRoot()
	assert.NotNil(t, root)

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
			expected: `INSERT INTO st_stock_stream ( tenant_code, remark, version, deleted ) VALUES (...)`,
		},
		{
			name:     "Select statement with include",
			id:       "selectByUuid",
			expected: `SELECT id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version FROM st_stock_stream WHERE 1=1 deleted = 0 AND uuid = ? AND warehouse_no = ? LIMIT 1`,
		},
	}

	fragments, err := xml.ExtractSQLFragments()
	assert.NoError(t, err)
	assert.Equal(t, "id, tenant_code, warehouse_no, uuid, business_type, business_no, remark, version", fragments["queryColumns"])

	analyzer := NewSQLAnalyzer()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			elem := root.FindElement(".//*[@id='" + tc.id + "']")
			assert.NotNil(t, elem)

			sql := xml.extractRawSQL(elem)
			preprocessedSQL := preprocessMyBatisXML(sql, fragments)
			t.Logf("Original SQL:\n%s\nPreprocessed SQL:\n%s", sql, preprocessedSQL)
			assert.Equal(t, tc.expected, strings.TrimSpace(preprocessedSQL))

			analyzer.AnalyzeStmt(root, "fn", tc.id, sql, fragments)
		})
	}
}
