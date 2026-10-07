import pytest
from app.ut_practice import ValidationError, validate_user_registration

# =====================================================================
# BLOCK 1: Data Validation & String Processing
# ---------------------------------------------------------------------
# Testing Guidance:
# - Implement both positive and negative test scenarios.
# - Use 'pytest.mark.parametrize' to efficiently cover multiple boundary values.
# - Validate specific error messages caught by 'pytest.raises(ValidationError)'.
# =====================================================================

@pytest.mark.parametrize("username, email, age", [
        ("kate", "test@gmail.com", 18)
    ])
def test_validate_user_registration_basic(username, email, age):
    assert validate_user_registration(username, email, age) is True

@pytest.mark.parametrize("username, email, age", [
        ("ok", "test1@gmail.com", 19,),  # when username is less than 3 characters
        ("okhsjhfsdjhsfdhkhfsahfds", "test2@gmail.com", 20),  # when username is more than 15 characters
        (" ", "test@gmail.com", 21),  # when having spaces in username
        ("", "test1@gmail.com", 22),  # when username is empty
        ("!@#$%^&", "test2@gmail.com", 23) # when username has special characters

    ])

def test_validate_user_registration_with_invalid_username(username, email, age):
    with pytest.raises(ValidationError, match="Username must be alphanumeric and between 3 and 15 characters."):
        validate_user_registration(username, email, age)

@pytest.mark.parametrize("username, email, age", [
        ("user1", "testgmail.com", 21),  # when email has no @
        ("user2", "", 22),  # when email is empty
        ("user3", "tes@t@gmail", 23) #when email has several @

    ])

def test_validate_user_registration_with_invalid_email(username, email, age):
    with pytest.raises(ValidationError, match="Invalid email format: missing or multiple '@'."):
        validate_user_registration(username, email, age)

@pytest.mark.parametrize("username, email, age", [
    ("user4", "test@gmail", 24), # when email has no domain part
    ("user5", "test@gmailcom", 25),  # when email has no dot at domain part
    ("user6", "test@.com", 26),  # when domain begins with dot in email
    ("user7", "test@gmail.com.", 26)  # when domain ends with dot in email

])

def test_validate_user_registration_with_invalid_domain_structure(username, email, age):
    with pytest.raises(ValidationError, match="Invalid email format: domain must contain a valid dot structure."):
        validate_user_registration(username, email, age)

@pytest.mark.parametrize("username, email, age", [
    ("user8", "test4@gmail.com", 17), # age is under 18
    ("user9", "test4@gmail.com", 100),  # age is more than 99
    ("user10", "test4@gmail.com", 18.5),  #  age is of a float format
    ("user11", "test4@gmail.com", "55")  # age is of a string format
])

def test_validate_user_registration_with_invalid_age(username, email, age):
    with pytest.raises(ValidationError, match="User must be between 18 and 99 years old."):
        validate_user_registration(username, email, age)


# unittest.TestCase does not support @pytest.mark.parametrize
# No class is required here
