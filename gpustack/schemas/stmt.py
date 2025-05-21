worker_after_drop_view_stmt_sqlite = "DROP VIEW IF EXISTS gpu_devices_view"
worker_after_create_view_stmt_sqlite = """
CREATE VIEW IF NOT EXISTS gpu_devices_view AS
SELECT
    w.name || ':' || json_extract(value, '$.type') || ':' || json_extract(value, '$.index') AS id,
    w.id as worker_id,
    w.name as worker_name,
    w.ip as worker_ip,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    json_extract(value, '$.uuid') AS uuid,
    json_extract(value, '$.name') AS name,
    json_extract(value, '$.vendor') AS vendor,
    json_extract(value, '$.index') AS 'index',
    json_extract(value, '$.device_index') AS 'device_index',
    json_extract(value, '$.device_chip_index') AS 'device_chip_index',
    json_extract(value, '$.core') AS core,
    json_extract(value, '$.memory') AS memory,
    json_extract(value, '$.network') AS network,
    json_extract(value, '$.temperature') AS temperature,
    json_extract(value, '$.labels') AS labels,
    json_extract(value, '$.type') AS type
FROM
    workers w,
    json_each(w.status, '$.gpu_devices')
WHERE
    json_array_length(w.status, '$.gpu_devices') > 0
"""

worker_after_drop_view_stmt_mysql = "DROP VIEW IF EXISTS gpu_devices_view"
worker_after_create_view_stmt_mysql = """
CREATE VIEW gpu_devices_view AS
SELECT
    CONCAT(w.name, ':', JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.type')), ':', JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.index'))) AS id,
    w.id AS worker_id,
    w.name AS worker_name,
    w.ip AS worker_ip,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.uuid')) AS uuid,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.name')) AS name,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.vendor')) AS vendor,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.index')) AS UNSIGNED) AS `index`,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.device_index')) AS UNSIGNED) AS `device_index`,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.device_chip_index')) AS UNSIGNED) AS `device_chip_index`,
    JSON_EXTRACT(gpu_device, '$.core') AS core,
    JSON_EXTRACT(gpu_device, '$.memory') AS memory,
    JSON_EXTRACT(gpu_device, '$.network') AS network,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.temperature')) AS DECIMAL(10, 2)) AS temperature,
    JSON_EXTRACT(gpu_device, '$.labels') AS labels,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.type')) AS type
FROM
    workers w,
    JSON_TABLE(w.status, '$.gpu_devices[*]' COLUMNS(
        gpu_device JSON PATH '$'
    )) AS gpu_devices
WHERE
    JSON_LENGTH(w.status, '$.gpu_devices') > 0
"""

worker_after_drop_view_stmt_postgres = "DROP VIEW IF EXISTS gpu_devices_view CASCADE"
worker_after_create_view_stmt_postgres = """
CREATE VIEW gpu_devices_view AS
SELECT
    w.name || ':' || (gpu_device::json->>'type') || ':' || (gpu_device::json->>'index') AS id,
    w.id as worker_id,
    w.name as worker_name,
    w.ip as worker_ip,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    (gpu_device::json->>'uuid') AS uuid,
    (gpu_device::json->>'name') AS name,
    (gpu_device::json->>'vendor') AS vendor,
    (gpu_device::json->>'index')::INTEGER AS "index",
    (gpu_device::json->>'device_index')::INTEGER AS "device_index",
    (gpu_device::json->>'device_chip_index')::INTEGER AS "device_chip_index",
    (gpu_device::json->>'core')::JSONB AS core,
    (gpu_device::json->>'memory')::JSONB AS memory,
    (gpu_device::json->>'network')::JSONB AS network,
    (gpu_device::json->>'temperature')::FLOAT AS temperature,
    (gpu_device::json->>'labels')::JSONB AS labels,
    (gpu_device::json->>'type') AS type
FROM
    workers w,
    LATERAL json_array_elements(w.status::json->'gpu_devices') AS gpu_device
WHERE
    json_array_length(w.status::json->'gpu_devices') > 0
"""
