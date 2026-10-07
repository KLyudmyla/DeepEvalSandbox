import pytest
import re
from app.ut_practice import FileUploadValidationError, FileUploadValidator

# =====================================================================
# BLOCK 5: Advanced Business Validation (File Upload Rules)
# ---------------------------------------------------------------------
# Testing Guidance:
# - Practice structural mapping testing by passing arrays of complex dictionaries.
# - Verify cascading limits: trigger file count limit first, then validation extension, then size limits.
# - Apply 'pytest.fixture' to pass pre-configured clean state files into your test cases.
# =====================================================================

@pytest.fixture
def my_files():
    return [
        {"name": "document.pdf", "size_mb": 2.5},
        {"name": "document.txt", "size_mb": 3},
        {"name": "document.docx", "size_mb": 4.5}
    ]

@pytest.fixture
def too_many_files():
    return [
        {"name": "document.pdf", "size_mb": 2.5},
        {"name": "document.txt", "size_mb": 3},
        {"name": "document.docx", "size_mb": 4.5},
        {"name": "document1.pdf", "size_mb": 2},
        {"name": "document2.pdf", "size_mb": 5},
        {"name": "document1.txt", "size_mb": 6}
    ]

@pytest.fixture
def wrong_format_files():
    return [
        {"name": "picture.png", "size_mb": 2.5},
        {"name": "map.html", "size_mb": 3},
    ]

@pytest.fixture
def large_file():
    return [
        {"name": "document123.pdf", "size_mb": 20},
        {"name": "contract.docx", "size_mb": 3},
    ]

@pytest.fixture
def file_with_missing_key():
    return [
        {"name": "document123.pdf"},
        {"name": "contract.docx", "size_mb": 3},
    ]

class TestFileUploadValidator:

    def test_validate_file_upload_basic(self, my_files):
        validator = FileUploadValidator(max_files=3, max_total_size_mb=10.0)
        assert validator.validate_batch(my_files) is True

    def test_validate_file_with_missing_key(self, file_with_missing_key):
        validator = FileUploadValidator(max_files=3, max_total_size_mb=10.0)
        assert validator.validate_batch(file_with_missing_key) is True

    def test_raising_exception_for_too_many_files(self, too_many_files):
        validator = FileUploadValidator(max_files=3, max_total_size_mb=10.0)
        with pytest.raises(FileUploadValidationError, match="Too many files. Maximum allowed is 3."):
            validator.validate_batch(too_many_files)

    def test_raising_exception_for_wrong_file_format(self, wrong_format_files):
        validator = FileUploadValidator(max_files=3, max_total_size_mb=10.0)
        with pytest.raises(FileUploadValidationError, match="File 'picture.png' has an unallowed format."):
            validator.validate_batch(wrong_format_files)

    def test_raising_exception_for_large_file(self, large_file):
        validator = FileUploadValidator(max_files=3, max_total_size_mb=10.0)
        with pytest.raises(FileUploadValidationError, match=re.escape("Total batch size (23.0 MB) exceeds limit of 10.0 MB.")):
            validator.validate_batch(large_file)

