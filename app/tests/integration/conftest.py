import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.it_practice import DATABASE_URL, Base
from app.main import app


@pytest.fixture(autouse=True) # to clean db after each test, use autouse=True
def engine():
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()

@pytest.fixture
def db_session(engine):
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
