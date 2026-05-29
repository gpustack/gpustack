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

    ``UNION`` (not ``UNION ALL``): a user that is a direct member of an
    Org AND a member of a Group that's a member of the same Org would
    otherwise emit two ``(org_id, user_id)`` rows from branches 2 and 3,
    which downstream ``non_admin_user_models`` JOINs would multiply into
    duplicate model rows for that user.
    """
    return '''
CREATE VIEW principal_users AS
SELECT u.id AS principal_id, u.id AS user_id
FROM principals u
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
UNION
SELECT pm.parent_principal_id AS principal_id, pm.member_principal_id AS user_id
FROM principal_memberships pm
JOIN principals u ON u.id = pm.member_principal_id
JOIN principals pr ON pr.id = pm.parent_principal_id
WHERE pm.deleted_at IS NULL
  AND pr.deleted_at IS NULL
  AND u.deleted_at IS NULL
  AND u.kind = 'USER'
  AND pr.kind IN ('ORG', 'GROUP')
UNION
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
    # MySQL's CAST(... AS TEXT) is invalid; CHAR is the canonical
    # variable-length text target there. PG / openGauss / sqlite all
    # accept TEXT.
    text_type = "CHAR" if db_type == "mysql" else "TEXT"
    # MySQL only accepts ``INTEGER`` as a CAST target from 8.0.17; older
    # servers / OceanBase compat modes want ``SIGNED``. PG / openGauss
    # both accept ``INTEGER``.
    int_type = "SIGNED" if db_type == "mysql" else "INTEGER"

    def _pid(via_expr: str) -> str:
        # ``pid`` keys identity-map lookups on MyModel, so it has to be
        # genuinely unique per emitted row. ``route_id:user_id`` alone
        # collides whenever a (user, route) is covered by multiple
        # grant chains; folding ``via_principal_id`` into the suffix
        # restores uniqueness. ``IFNULL`` / ``COALESCE`` are mandatory:
        # both ``CONCAT(...)`` (MySQL) and ``||`` (PG / openGauss /
        # sqlite) propagate NULL through the whole expression, so a raw
        # NULL via on the PUBLIC/AUTHED branch would make pid itself
        # NULL.
        if db_type == "mysql":
            return (
                f"CONCAT(m.id, ':', u.id, ':', "
                f"IFNULL(CAST({via_expr} AS CHAR), ''))"
            )
        return (
            f"CAST(m.id AS TEXT) || ':' || CAST(u.id AS TEXT) "
            f"|| ':' || COALESCE(CAST({via_expr} AS TEXT), '')"
        )

    # 2-branch UNION ALL — each branch is a straight index join, so the
    # planner doesn't have to materialize every (user, route) pair to
    # then OR-filter EXISTS subqueries against it. ``mrp.deleted_at IS
    # NULL`` is required on the ACL branch: leaving it off was the
    # soft-delete-leak bug from review.
    # After identity consolidation, USER rows live in ``principals``
    # (kind = 'USER'). The single ACL branch joins through
    # ``principal_users``, which maps a USER-principal to itself — so a
    # USER grant resolves exactly like the released ``allowed_users``
    # policy did (now folded into ALLOWED_PRINCIPALS; the migration
    # converted existing rows), and that separate branch is gone.
    #
    # ``via_principal_id`` / ``via_principal_kind`` record which principal
    # granted this (user, route) row visibility, so the API layer can
    # partition results into Personal vs Org-scoped views without
    # re-deriving the chain. NULL on the PUBLIC/AUTHED branch — those
    # grants aren't principal-mediated. The kind is normalized to TEXT
    # so PG's native ``principaltype`` enum doesn't poison the UNION's
    # column type inference.
    #
    # The view's contract is one row per chain — multi-chain (user,
    # route) collapse to "one row per route" is the API layer's job
    # (see ``_get_model_routes``), because the visibility filter
    # discriminates by chain ``via_principal_kind`` and would drop the
    # wrong chain if the view picked one up front.
    return f'''
CREATE VIEW non_admin_user_models AS
SELECT {_pid("NULL")} AS pid,
       u.id AS user_id,
       CAST(NULL AS {int_type}) AS via_principal_id,
       CAST(NULL AS {text_type}) AS via_principal_kind,
       m.*
FROM principals u
CROSS JOIN model_routes m
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
  AND u.is_admin = {sql_false}
  AND m.access_policy IN ('PUBLIC', 'AUTHED')

UNION ALL

SELECT {_pid("mrp.principal_id")} AS pid,
       u.id AS user_id,
       mrp.principal_id AS via_principal_id,
       CAST(pr.kind AS {text_type}) AS via_principal_kind,
       m.*
FROM principals u
JOIN principal_users pu ON pu.user_id = u.id
JOIN model_route_principals mrp
  ON mrp.principal_id = pu.principal_id
  AND mrp.deleted_at IS NULL
JOIN principals pr
  ON pr.id = mrp.principal_id
JOIN model_routes m
  ON m.id = mrp.route_id
  AND m.access_policy = 'ALLOWED_PRINCIPALS'
WHERE u.kind = 'USER' AND u.deleted_at IS NULL
  AND u.is_admin = {sql_false}
'''
