"""Tests for difficulty classification from model pass counts."""


from arxiv_manager.tracking import classify_difficulty


class TestClassifyDifficulty:
    def test_qwen_pass_any_means_easy(self):
        """Qwen passing any run forces EASY regardless of Gemini."""
        assert classify_difficulty(1, 0) == "easy"
        assert classify_difficulty(4, 0) == "easy"
        assert classify_difficulty(1, 4) == "easy"
        assert classify_difficulty(3, 2) == "easy"

    def test_qwen_fails_all_gemini_passes_means_challenging(self):
        """Qwen fails all, Gemini passes any → CHALLENGING."""
        assert classify_difficulty(0, 1) == "challenging"
        assert classify_difficulty(0, 4) == "challenging"

    def test_both_fail_all_means_hardest(self):
        """Both models failing every run → HARDEST."""
        assert classify_difficulty(0, 0) == "hardest"

    def test_partial_counts(self):
        """Partial pass counts follow the same rules."""
        assert classify_difficulty(2, 2) == "easy"
        assert classify_difficulty(0, 3) == "challenging"
