worker_after_drop_view_stmt_sqlite = "DROP VIEW IF EXISTS gpu_devices_view"
worker_after_create_view_stmt_sqlite = """
CREATE VIEW IF NOT EXISTS gpu_devices_view AS
SELECT
    w.name || ':' || json_extract(value, '$.type') || ':' || json_extract(value, '$.index') AS 'id',
    w.id AS 'worker_id',
    w.name AS 'worker_name',
    w.ip AS 'worker_ip',
    w.ifname AS 'worker_ifname',
    w.cluster_id,
    w.owner_principal_id,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    json_extract(value, '$.vendor') AS 'vendor',
    json_extract(value, '$.type') AS 'type',
    json_extract(value, '$.index') AS 'index',
    json_extract(value, '$.device_index') AS 'device_index',
    json_extract(value, '$.device_chip_index') AS 'device_chip_index',
    json_extract(value, '$.arch_family') AS 'arch_family',
    json_extract(value, '$.name') AS 'name',
    json_extract(value, '$.uuid') AS 'uuid',
    json_extract(value, '$.driver_version') AS 'driver_version',
    json_extract(value, '$.runtime_version') AS 'runtime_version',
    json_extract(value, '$.compute_capability') AS 'compute_capability',
    json_extract(value, '$.core') AS 'core',
    json_extract(value, '$.memory') AS 'memory',
    json_extract(value, '$.temperature') AS 'temperature',
    json_extract(value, '$.network') AS 'network'
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
    CONCAT(w.name, ':', JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.type')), ':', JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.index'))) AS `id`,
    w.id AS `worker_id`,
    w.name AS `worker_name`,
    w.ip AS `worker_ip`,
    w.ifname AS `worker_ifname`,
    w.cluster_id,
    w.owner_principal_id,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.vendor')) AS `vendor`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.type')) AS `type`,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.index')) AS UNSIGNED) AS `index`,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.device_index')) AS UNSIGNED) AS `device_index`,
    CAST(JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.device_chip_index')) AS UNSIGNED) AS `device_chip_index`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.arch_family')) AS `arch_family`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.name')) AS `name`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.uuid')) AS `uuid`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.driver_version')) AS `driver_version`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.runtime_version')) AS `runtime_version`,
    JSON_UNQUOTE(JSON_EXTRACT(gpu_device, '$.compute_capability')) AS `compute_capability`,
    JSON_EXTRACT(gpu_device, '$.core') AS `core`,
    JSON_EXTRACT(gpu_device, '$.memory') AS `memory`,
    CAST(COALESCE(JSON_VALUE(gpu_device, '$.temperature'), '0') AS DECIMAL(10, 2)) AS `temperature`,
    JSON_EXTRACT(gpu_device, '$.network') AS `network`
FROM
    workers w,
    JSON_TABLE(w.status, '$.gpu_devices[*]' COLUMNS(
        gpu_device JSON PATH '$'
    )) AS gpu_devices
WHERE
    JSON_LENGTH(w.status, '$.gpu_devices') IS NOT NULL
    AND JSON_LENGTH(w.status, '$.gpu_devices') > 0
"""

worker_after_drop_view_stmt_postgres = "DROP VIEW IF EXISTS gpu_devices_view CASCADE"
worker_after_create_view_stmt_postgres = """
CREATE VIEW gpu_devices_view AS
SELECT
    w.name || ':' || (gpu_device::json->>'type') || ':' || (gpu_device::json->>'index') AS "id",
    w.id AS "worker_id",
    w.name AS "worker_name",
    w.ip AS "worker_ip",
    w.ifname AS "worker_ifname",
    w.cluster_id,
    w.owner_principal_id,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    (gpu_device::json->>'vendor') AS "vendor",
    (gpu_device::json->>'type') AS "type",
    (gpu_device::json->>'index')::INTEGER AS "index",
    (gpu_device::json->>'device_index')::INTEGER AS "device_index",
    (gpu_device::json->>'device_chip_index')::INTEGER AS "device_chip_index",
    (gpu_device::json->>'arch_family') AS "arch_family",
    (gpu_device::json->>'name') AS "name",
    (gpu_device::json->>'uuid') AS "uuid",
    (gpu_device::json->>'driver_version') AS "driver_version",
    (gpu_device::json->>'runtime_version') AS "runtime_version",
    (gpu_device::json->>'compute_capability') AS "compute_capability",
    (gpu_device::json->>'core')::JSONB AS "core",
    (gpu_device::json->>'memory')::JSONB AS "memory",
    (gpu_device::json->>'temperature')::FLOAT AS "temperature",
    (gpu_device::json->>'network')::JSONB AS "network"
FROM
    workers w,
    LATERAL json_array_elements(w.status::json->'gpu_devices') AS gpu_device
WHERE
    json_typeof(w.status::json->'gpu_devices') = 'array';
"""

# openGauss does not support json_array_elements (added in PostgreSQL 9.3).
# Use generate_series + integer-index -> operator to expand the JSONB array instead.
worker_after_create_view_stmt_opengauss = """
CREATE VIEW gpu_devices_view AS
SELECT
    w.name || ':' || (w.status::jsonb->'gpu_devices'->s.idx->>'type') || ':' || (w.status::jsonb->'gpu_devices'->s.idx->>'index') AS "id",
    w.id AS "worker_id",
    w.name AS "worker_name",
    w.ip AS "worker_ip",
    w.ifname AS "worker_ifname",
    w.cluster_id,
    w.owner_principal_id,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    (w.status::jsonb->'gpu_devices'->s.idx->>'vendor') AS "vendor",
    (w.status::jsonb->'gpu_devices'->s.idx->>'type') AS "type",
    (w.status::jsonb->'gpu_devices'->s.idx->>'index')::INTEGER AS "index",
    (w.status::jsonb->'gpu_devices'->s.idx->>'device_index')::INTEGER AS "device_index",
    (w.status::jsonb->'gpu_devices'->s.idx->>'device_chip_index')::INTEGER AS "device_chip_index",
    (w.status::jsonb->'gpu_devices'->s.idx->>'arch_family') AS "arch_family",
    (w.status::jsonb->'gpu_devices'->s.idx->>'name') AS "name",
    (w.status::jsonb->'gpu_devices'->s.idx->>'uuid') AS "uuid",
    (w.status::jsonb->'gpu_devices'->s.idx->>'driver_version') AS "driver_version",
    (w.status::jsonb->'gpu_devices'->s.idx->>'runtime_version') AS "runtime_version",
    (w.status::jsonb->'gpu_devices'->s.idx->>'compute_capability') AS "compute_capability",
    (w.status::jsonb->'gpu_devices'->s.idx->'core')::JSONB AS "core",
    (w.status::jsonb->'gpu_devices'->s.idx->'memory')::JSONB AS "memory",
    (w.status::jsonb->'gpu_devices'->s.idx->>'temperature')::FLOAT AS "temperature",
    (w.status::jsonb->'gpu_devices'->s.idx->'network')::JSONB AS "network"
FROM
    workers w,
    generate_series(0, jsonb_array_length(w.status::jsonb->'gpu_devices') - 1) AS s(idx)
WHERE
    jsonb_typeof(w.status::jsonb->'gpu_devices') = 'array';
"""

model_user_after_drop_view_stmt = "DROP VIEW IF EXISTS non_admin_user_models"
principal_users_after_drop_view_stmt = "DROP VIEW IF EXISTS principal_users"


def principal_users_after_create_view_stmt() -> str:
    """Helper view: (principal_id, user_id) — every user covered by a
    principal, expanded across direct USER ownership, direct ORG/GROUP
    membership, and transitive ORG membership via a joined Group.
    Used by ``non_admin_user_models`` so the ALLOWED_PRINCIPALS branch
    can index-join instead of running a correlated EXISTS over
    ``principal_memberships`` per row.

    After identity consolidation, USER rows live in the same
    ``principals`` table as ORG / GROUP — a USER-principal's id IS the
    user's id, so the first branch is a self-trivial select instead of
    the old ``users JOIN principals`` shape.

    Three branches:

    1. The user themselves (always covered by their USER-principal).
    2. Direct: ``(parent=Org/Group, member=User)`` — user joined the
       Org/Group directly.
    3. Transitive: user is in a Group that is itself a member of an
       Org — propagates Org membership to every active user in the
       Group.
    """
    return '''
CREATE VIEW principal_users AS
SELECT u.id AS principal_id, u.id AS user_id
FROM principals u
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
UNION ALL
SELECT pm.parent_principal_id AS principal_id, pm.member_principal_id AS user_id
FROM principal_memberships pm
JOIN principals u ON u.id = pm.member_principal_id
JOIN principals pr ON pr.id = pm.parent_principal_id
WHERE pm.deleted_at IS NULL
  AND pr.deleted_at IS NULL
  AND u.deleted_at IS NULL
  AND u.kind = 'USER'
  AND pr.kind IN ('ORG', 'GROUP')
UNION ALL
SELECT org_pm.parent_principal_id AS principal_id, group_pm.member_principal_id AS user_id
FROM principal_memberships group_pm
JOIN principal_memberships org_pm
  ON org_pm.member_principal_id = group_pm.parent_principal_id
 AND org_pm.deleted_at IS NULL
JOIN principals u ON u.id = group_pm.member_principal_id
JOIN principals grp ON grp.id = group_pm.parent_principal_id
JOIN principals org ON org.id = org_pm.parent_principal_id
WHERE group_pm.deleted_at IS NULL
  AND grp.deleted_at IS NULL
  AND grp.kind = 'GROUP'
  AND org.deleted_at IS NULL
  AND org.kind = 'ORG'
  AND u.kind = 'USER'
  AND u.deleted_at IS NULL
'''


def model_user_after_create_view_stmt(db_type: str) -> str:
    sql_false = '0' if db_type == "sqlite" else 'FALSE'
    pid = (
        "CONCAT(m.id, ':', u.id)"
        if db_type == "mysql"
        else "CAST(m.id AS TEXT) || ':' || CAST(u.id AS TEXT)"
    )
    # 4-branch UNION ALL — each branch is a straight index join, so the
    # planner doesn't have to materialize every (user, route) pair to
    # then OR-filter EXISTS subqueries against it. ``mrp.deleted_at IS
    # NULL`` is required on every ACL branch: leaving it off was the
    # soft-delete-leak bug from review.
    # After identity consolidation, USER rows live in ``principals``
    # (kind = 'USER'). Every reference to the old ``users`` table is
    # rewritten to ``principals`` with a kind filter. The
    # ALLOWED_USERS branch becomes ``mrp.principal_id = u.id`` (the
    # user's USER-principal id IS the user's id) instead of joining
    # through the now-removed ``users.principal_id`` column.
    return f'''
CREATE VIEW non_admin_user_models AS
SELECT {pid} AS pid, u.id AS user_id, m.*
FROM principals u
CROSS JOIN model_routes m
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
  AND u.is_admin = {sql_false}
  AND m.access_policy IN ('PUBLIC', 'AUTHED')

UNION ALL

SELECT {pid} AS pid, u.id AS user_id, m.*
FROM principals u
JOIN principal_users pu
  ON pu.user_id = u.id
JOIN model_routes m
  ON m.owner_principal_id = pu.principal_id
  AND m.access_policy = 'ORG'
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
  AND u.is_admin = {sql_false}

UNION ALL

SELECT {pid} AS pid, u.id AS user_id, m.*
FROM principals u
JOIN model_route_principals mrp
  ON mrp.principal_id = u.id
  AND mrp.deleted_at IS NULL
JOIN model_routes m
  ON m.id = mrp.route_id
  AND m.access_policy = 'ALLOWED_USERS'
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
  AND u.is_admin = {sql_false}

UNION ALL

SELECT {pid} AS pid, u.id AS user_id, m.*
FROM principals u
JOIN principal_users pu ON pu.user_id = u.id
JOIN model_route_principals mrp
  ON mrp.principal_id = pu.principal_id
  AND mrp.deleted_at IS NULL
JOIN model_routes m
  ON m.id = mrp.route_id
  AND m.access_policy = 'ALLOWED_PRINCIPALS'
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
  AND u.is_admin = {sql_false}
'''
