"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_html():
    """Sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Article</title></head>
    <body>
        <article>
            <h1>The Future of AI</h1>
            <time datetime="2024-01-15">January 15, 2024</time>
            <p>This is the first paragraph of the article. It contains
            important information about AI developments.</p>
            <p>Here is another paragraph with more details about the topic.
            Technology is advancing rapidly.</p>
            <h2>Key Points</h2>
            <p>The main takeaway is that AI will transform how we work.</p>
        </article>
    </body>
    </html>
    """


@pytest.fixture
def sample_content():
    """Sample article content for testing."""
    return """
    The future of artificial intelligence is fascinating. We're seeing rapid
    developments in machine learning, natural language processing, and automation.

    Here's the thing: most people underestimate how quickly AI is advancing.
    The technology that seemed like science fiction five years ago is now
    becoming reality.

    You should pay attention to these trends:
    1. Large language models are becoming more capable
    2. AI tools are becoming more accessible
    3. The cost of AI is decreasing rapidly

    The question isn't whether AI will transform your industry - it's when.
    Those who adapt early will have a significant advantage.

    Take action today. Start learning about AI. Experiment with the tools.
    Build projects that solve real problems. The future belongs to those
    who prepare for it.
    """
