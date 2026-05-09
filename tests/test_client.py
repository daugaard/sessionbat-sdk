from __future__ import annotations

import unittest

from sessionbat import SessionBat


class SessionTest(unittest.TestCase):
    def test_does_not_expose_standalone_error_observation_api(self) -> None:
        session = SessionBat().session(session_id="thread_123")

        self.assertFalse(hasattr(session, "error"))


if __name__ == "__main__":
    unittest.main()
