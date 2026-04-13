from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector


def is_opengauss(conn) -> bool:
    """Check if the database is openGauss (presents as postgresql dialect).

    openGauss is a PostgreSQL-compatible database that lacks certain aggregate
    functions such as jsonb_agg and set-returning functions like
    jsonb_array_elements. Migrations that use those functions must detect this
    and fall back to Python-based processing instead.
    """
    if conn.dialect.name != 'postgresql':
        return False
    try:
        version = conn.execute(sa.text("SELECT version()")).scalar()
        return 'openGauss' in (version or '')
    except Exception:
        return False


def column_exists(table_name, column_name) -> bool:
    """Check if a column exists in a table.
    Args:
        table_name (str): The name of the table.
        column_name (str): The name of the column.
    Returns:
        bool: True if the column exists, False otherwise.
    """
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def table_exists(table_name) -> bool:
    """Check if a table exists in the database.
    Args:
        table_name (str): The name of the table."
    Returns:
        bool: True if the table exists, False otherwise.
    """
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    return table_name in inspector.get_table_names()
