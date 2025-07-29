from alembic import command
from alembic.config import Config as AlembicConfig
import importlib.util
import os,re

spec = importlib.util.find_spec("gpustack")
if spec is None:
    raise ImportError("The 'gpustack' package is not found.")

pkg_path = spec.submodule_search_locations[0]
alembic_cfg = AlembicConfig()
alembic_cfg.set_main_option(
    "script_location", os.path.join(pkg_path, "migrations")
)

db_url = "mysql://a_appconnect:lPjK:POUNm0Z@10.62.175.98:3306/gpustack2"
# Use the pymysql driver to execute migrations to avoid compatibility issues between asynchronous drivers and Alembic.
if db_url.startswith("mysql://"):
    db_url = re.sub(r'^mysql://', 'mysql+pymysql://', db_url)
db_url_escaped = db_url.replace("%", "%%")
alembic_cfg.set_main_option("sqlalchemy.url", db_url_escaped)
# try:
command.upgrade(alembic_cfg, "head")
# except Exception as e:
#     raise RuntimeError(f"Database migration failed: {e}") from e
