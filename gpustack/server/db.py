from sqlmodel import SQLModel, create_engine


_engine = None


def get_engine():
    return _engine


def init_db(db_url: str):
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False}
        _engine = create_engine(db_url, echo=False, connect_args=connect_args)
    create_db_and_tables(_engine)


def create_db_and_tables(engine):
    SQLModel.metadata.create_all(engine)
