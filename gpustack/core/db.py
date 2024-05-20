import secrets
from sqlmodel import SQLModel, Session, create_engine

from .config import configs
from ..logging import logger
from ..core.security import get_password_hash
from ..schemas.models import Model
from ..schemas.users import User, UserCreate


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False}
        _engine = create_engine(
            configs.database_url, echo=False, connect_args=connect_args
        )
    return _engine


def init_db():
    engine = get_engine()
    create_db_and_tables(engine)
    with Session(engine) as session:
        init_data(session)


def create_db_and_tables(engine):
    SQLModel.metadata.create_all(engine)


def init_data(session: Session):
    init_data_funcs = [init_model, init_user]
    for init_data_func in init_data_funcs:
        init_data_func(session)


def init_model(session: Session):
    if configs.model:
        huggingface_model_id = configs.model
    else:
        return

    model_name = huggingface_model_id.split("/")[-1]

    model = Model.first_by_field(session=session, field="name", value=model_name)
    if not model:
        model = Model(
            name=model_name,
            source="huggingface",
            huggingface_model_id=huggingface_model_id,
        )
        model.save(session)

        logger.info("Created model: %s", model_name)


def init_user(session: Session):
    user = User.first_by_field(session=session, field="name", value="admin")
    if not user:
        bootstrap_password = configs.bootstrap_password
        if not bootstrap_password:
            bootstrap_password = secrets.token_urlsafe(16)
            logger.info("!!!Bootstrap password!!!: %s", bootstrap_password)

        user_create = UserCreate(
            name="admin",
            full_name="System Admin",
            password=bootstrap_password,
            is_admin=True,
        )
        user = User.model_validate(
            user_create,
            update={"hashed_password": get_password_hash(user_create.password)},
        )
        user.save(session)
