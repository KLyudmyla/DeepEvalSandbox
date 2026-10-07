import pytest
from datetime import datetime
from app.ut_practice import SubscriptionExpiredError, SubscriptionManager

# =====================================================================
# BLOCK 3: Date, Time & Expiration Logic
# ---------------------------------------------------------------------
# Testing Guidance:
# - CRITICAL: Do NOT use dynamic time like 'datetime.now()' in tests.
# - Inject hardcoded 'current_time' values to make assertions entirely deterministic.
# - Test critical threshold transitions: exactly active, exactly inside grace period, and fully expired.
# =====================================================================

class TestSubscriptionManager:

    def test_verifying_access_with_active_subscription(self):
        output = SubscriptionManager("user1", "plan1", datetime(2027, 1, 1))
        current_time = datetime(2026, 6, 19, 16, 00, 00)
        assert output.verify_access(current_time) == "active"

    def test_verifying_access_within_grace_period(self):
        output = SubscriptionManager("user1", "plan1", datetime(2026, 6, 16))
        current_time = datetime(2026, 6, 18, 16, 00, 00)
        assert output.verify_access(current_time) == "grace_period"

    def test_raising_error_on_expired_subscription(self):
        output = SubscriptionManager("user1", "plan1", datetime(2020, 1, 1))
        current_time = datetime(2026, 6, 19, 16, 00, 00)
        with pytest.raises(SubscriptionExpiredError, match="Access denied. User subscription expired on 2020-01-01 00:00:00."):
            output.verify_access(current_time)
