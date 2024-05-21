from typing import Annotated
from fastapi import Depends
from sqlmodel import Session

from gpustack.server.db import get_engine
from gpustack.schemas.common import ListParams


def get_session():
    engine = get_engine()
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
ListParamsDep = Annotated[ListParams, Depends(ListParams)]
