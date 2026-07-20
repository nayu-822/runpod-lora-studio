from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from runpod_lora_studio.config.settings import AppSettings


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def create_session_factory(settings: AppSettings) -> sessionmaker[Session]:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(sqlite_url(settings.database_path), future=True)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_engine_for_settings(settings: AppSettings) -> Engine:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(sqlite_url(settings.database_path), future=True)


def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
