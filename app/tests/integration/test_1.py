from app.it_practice import User, Article
from app.tests.integration.conftest import client, db_session


def test_there_are_no_users(client, db_session):
    response = client.get("/users/")
    assert response.status_code == 200
    assert response.json()== []

    users_number = db_session.query(User).count()
    assert users_number == 0

def test_create_user(client, db_session):
    response = client.post(
        "/users/",
        params={"username": "alice3", "email": "alice3@example.com"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "alice3"
    user_id = data["id"]

    db_user = db_session.query(User).filter(User.id == user_id).first()
    assert db_user is not None
    assert db_user.username == "alice3"

def test_get_users(client, db_session):
    response = client.post(
        "/users/",
        params={"username": "alice4", "email": "alice4@example.com"}
    )
    assert response.status_code == 201

    response2 = client.get("/users/")
    assert response2.status_code == 200
    users_number = db_session.query(User).count()
    assert users_number == 1

def test_create_user_with_used_email(client, db_session):
    response = client.post(
        "/users/",
        params={"username": "alice8", "email": "alice8@example.com"}
    )
    assert response.status_code == 201
    user_number = db_session.query(User).count()
    assert user_number == 1

    response2 = client.post(
        "/users/",
        params={"username": "alice89", "email": "alice8@example.com"}
    )
    assert response2.status_code == 400
    data = response2.json()
    assert data["detail"] == "Email already registered"

    user_number1 = db_session.query(User).count()
    assert user_number1 == 1

def test_create_article(client, db_session):
    response = client.post(
        "/users/",
        params={"username": "kate", "email": "test@example.com"}
    )
    assert response.status_code == 201
    data = response.json()
    owner = data["id"]
    user_count = db_session.query(User).count()
    assert user_count == 1

    response2 = client.post(
        "/articles/",
        params={"title": "test", "content": "test message", "owner_id": owner}
    )
    assert response2.status_code == 201
    article_info = db_session.query(Article).filter(Article.title == "test").first()
    assert article_info is not None
    assert article_info.title == "test"

def test_non_existing_user_creates_article(client, db_session):
    response = client.post(
        "/articles/",
        params={"title": "test", "content": "test", "owner_id": 1}
    )
    assert response.status_code == 404
    data= response.json()
    assert data["detail"] == "User not found"

    article_number = db_session.query(Article).count()
    assert article_number == 0

def test_no_articles(client, db_session):
    response = client.get("/articles/")
    assert response.status_code == 200
    assert response.json() == []

    articles_number = db_session.query(Article).count()
    assert articles_number == 0

def test_get_articles(client, db_session):
    response = client.post(
        "/users/",
        params={"username": "superman", "email": "superman@example.com"}
    )
    assert response.status_code == 201
    data = response.json()
    owner = data["id"]
    user_count = db_session.query(User).count()
    assert user_count == 1

    response2 = client.post(
        "/articles/",
        params={"title": "my 1st article", "content": "test message", "owner_id": owner}
    )
    assert response2.status_code == 201
    article_info = db_session.query(Article).filter(Article.title == "my 1st article").first()
    assert article_info is not None
    assert article_info.title == "my 1st article"

    response3 = client.get(
        "/articles/",
    )
    assert response3.status_code == 200

    articles_number = db_session.query(Article).count()
    assert articles_number == 1
