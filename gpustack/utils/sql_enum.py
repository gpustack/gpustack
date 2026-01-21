from typing import Dict, Tuple, List

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import Connection


def add_enum_values(
    table_columns: Dict[str, str], original_enum: sa.Enum, *to_add_values: str
):
    """
    add_enum_values add new values to an existing enum type in the database.

    :param table_columns: a dictionary mapping table names to their column definitions
    :type table_columns: Dict[str, str]
    :param original_enum: existing enum type in the database
    :type original_enum: sa.Enum
    :param to_add_values: new values to add to the enum
    :type to_add_values: Tuple[str, ...]
    """
    if len(to_add_values) == 0:
        return
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        for value in to_add_values:
            conn.execute(
                sa.text(f"ALTER TYPE {original_enum.name} ADD VALUE '{value}'")
            )
    elif conn.dialect.name == 'mysql':
        add_mysql_enum_values(table_columns, *to_add_values)


def add_mysql_enum_values(table_columns: Dict[str, str], *to_add_values: str):
    conn = op.get_bind()
    for table_name, column_name in table_columns.items():
        modify_mysql_table_column_enum(
            conn, table_name, column_name, list(to_add_values), []
        )


def modify_mysql_table_column_enum(
    conn: Connection,
    table_name: str,
    column_name: str,
    to_add_values: List[str],
    to_remove_values: List[str],
):
    result = conn.execute(
        sa.text(
            f"""
            SELECT COLUMN_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            AND COLUMN_NAME = '{column_name}'
            AND TABLE_SCHEMA = DATABASE()
        """
        )
    ).scalar()

    existing_values = []
    if result:
        enum_str = result.split("enum(")[1].split(")")[0]
        existing_values = [v.strip("'") for v in enum_str.split("','")]

    new_values = [v for v in existing_values if v not in to_remove_values]
    new_values.extend(to_add_values)
    if set(new_values) != set(existing_values):
        new_enum_str = "enum('" + "','".join(new_values) + "')"

        # Construct new ALTER TABLE statement
        alter_sql = (
            f"ALTER TABLE {table_name} MODIFY COLUMN {column_name} {new_enum_str};"
        )

        # Execute modification
        conn.execute(sa.text(alter_sql))


def remove_postgres_enum_values(
    conn: Connection,
    table_name: str,
    column_name: str,
    original_enum: sa.Enum,
    *to_remove_values: str,
):
    new_enum_values_str = ','.join(
        [repr(v) for v in original_enum.enums if v not in to_remove_values]
    )
    conn.execute(
        sa.text(f"CREATE TYPE {original_enum.name}tmp AS ENUM ({new_enum_values_str});")
    )
    conn.execute(
        sa.text(
            f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {original_enum.name}tmp USING {column_name}::text::{original_enum.name}tmp;"
        )
    )
    conn.execute(sa.text(f"DROP TYPE {original_enum.name};"))
    conn.execute(
        sa.text(f"ALTER TYPE {original_enum.name}tmp RENAME TO {original_enum.name};")
    )


def remove_enum_values(
    table_columns: Dict[str, Tuple[str, str]],
    original_enum: sa.Enum,
    *to_remove_values: str,
):
    """
    remove_enum_values removes specified values from an existing enum type in the database.

    :param table_columns: a dictionary mapping table names to their column definitions
    :type table_columns: Dict[str, Tuple[str, str]]
    :param original_enum: existing enum type in the database
    :type original_enum: sa.Enum
    :param to_remove_values: values to remove from the enum
    :type to_remove_values: Tuple[str, ...]
    """
    if len(to_remove_values) == 0:
        return
    conn = op.get_bind()
    for table_name, (column_name, default_value) in table_columns.items():
        conn.execute(
            sa.text(
                f"""
            UPDATE {table_name}
            SET {column_name} = {repr(default_value)}
            WHERE {column_name} IN ({','.join([repr(v) for v in to_remove_values])});
        """
            )
        )
        if conn.dialect.name == 'mysql':
            modify_mysql_table_column_enum(
                conn, table_name, column_name, [], list(to_remove_values)
            )
        if conn.dialect.name == 'postgresql':
            remove_postgres_enum_values(
                conn, table_name, column_name, original_enum, *to_remove_values
            )
