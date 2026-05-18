from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from services.config import DatabaseConfig
from services.database.models import Base

class DatabaseSessionManager:
    def __init__(self, config: DatabaseConfig):
        self.enabled = config.enabled
        self.create_tables_startup = config.create_tables
        self.engine = None
        self.session_factory = None

        if not self.enabled:
            print("Database initialization is disabled in the configuration.")
            return
        
        sql_config = {"echo": config.echo, "future": True}
        sql_config["pool_size"] = config.pool_size
        sql_config["max_overflow"] = config.max_overflow

        self.engine = create_engine(config.url, **sql_config)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        if self.engine is not None and self.create_tables_startup:
            Base.metadata.create_all(self.engine)
    
    @contextmanager
    def session(self):
        if self.session_factory is None:
            yield None
            return

        db: Session = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
