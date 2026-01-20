# MindForge AI - Dan Koe Style Content Generation System

A comprehensive AI system for analyzing writing styles, extracting thinking patterns, and generating authentic content inspired by Dan Koe's approach to tech analysis.

## Features

- **Article Crawling**: Automatically collect articles from Dan Koe's newsletter
- **Style Analysis**: Extract vocabulary, sentence structure, and rhetorical patterns
- **Thinking Pattern Extraction**: Identify and codify mental models and argument structures
- **Web Research Engine**: Topic-specific research with credibility weighting
- **Cognitive Processing**: Apply thinking frameworks to generate insights
- **Style Generation**: Generate articles that match the target style
- **Bilingual Support**: English and Simplified Chinese output
- **Evaluation System**: Quantify similarity, creativity, and human-likeness
- **Interactive Assistant**: Real-time collaborative writing with style evolution tracking

## Installation

### Prerequisites

- Python 3.10+
- Poetry (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mindforge-agent.git
cd mindforge-agent
```

2. Install dependencies with Poetry:
```bash
poetry install
```

Or with pip:
```bash
pip install -e .
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Configure your API keys in `.env`:
```env
DEEPSEEK_API_KEY=your_key_here
# Or use OpenAI/Anthropic
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional: For web search
SERPER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

5. (Optional) Download spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

## Usage

### CLI Commands

```bash
# Check system status
mindforge status

# Crawl articles from Dan Koe's newsletter
mindforge crawl --max 50

# Process crawled articles
mindforge process

# Analyze style and thinking patterns
mindforge analyze

# Generate an article
mindforge generate "AI agents and the future of work"

# Generate in Chinese
mindforge generate "AI agents" --lang zh

# Evaluate a generated article
mindforge evaluate output/article.json --detailed

# Start interactive writing session
mindforge interactive
```

### Python API

```python
from mindforge_dankoe.crawler import DanKoeSpider, ArticleProcessor
from mindforge_dankoe.analysis import StyleAnalyzer, ThinkingExtractor
from mindforge_dankoe.generation import ResearchEngine, CognitiveProcessor, StyleGenerator
from mindforge_dankoe.evaluation import SimilarityMetrics, CreativityScorer

# Crawl articles
spider = DanKoeSpider(max_articles=50)
articles = await spider.crawl_all()

# Process articles
processor = ArticleProcessor()
processed = processor.process_all(articles)

# Create style fingerprint
analyzer = StyleAnalyzer()
fingerprint = analyzer.create_fingerprint(processed)

# Extract thinking patterns
extractor = ThinkingExtractor()
for article in processed:
    extractor.analyze_article(article)
patterns = extractor.synthesize_patterns(extractor.extracted_analyses)

# Generate new content
research_engine = ResearchEngine()
research = await research_engine.research_topic("AI agents")

cognitive = CognitiveProcessor()
insights = cognitive.apply_thinking_template("AI agents", research)

generator = StyleGenerator(style_fingerprint=fingerprint)
article = generator.generate_article(insights, target_language="en")

# Evaluate output
similarity = SimilarityMetrics(style_fingerprint=fingerprint)
scores = similarity.calculate_similarity(article)
print(f"Style similarity: {scores.overall_score:.1%}")
```

## Project Structure

```
mindforge_dankoe/
├── crawler/                    # Article collection
│   ├── dankoe_spider.py       # Web crawler
│   └── article_processor.py   # Content processing
├── analysis/                   # Pattern extraction
│   ├── thinking_extractor.py  # Thinking frameworks
│   ├── style_analyzer.py      # Style fingerprinting
│   └── bilingual_handler.py   # Chinese-English handling
├── generation/                 # Content creation
│   ├── research_engine.py     # Web research
│   ├── cognitive_processor.py # Thinking application
│   └── style_generator.py     # Article generation
├── evaluation/                 # Quality assessment
│   ├── similarity_metrics.py  # Style matching
│   ├── creativity_scorer.py   # Creativity quantification
│   └── human_likeness_tests.py# Authenticity testing
├── config/                     # Configuration
│   └── settings.py            # Application settings
├── utils/                      # Utilities
│   ├── logger.py              # Logging
│   └── text_processing.py     # Text utilities
├── cli.py                      # Command line interface
└── interactive.py              # Interactive assistant
```

## Evaluation Metrics

### Style Similarity
- Thinking pattern matching
- Vocabulary similarity
- Sentence structure alignment
- Tone consistency

### Creativity Scoring
- **Novelty**: New insights not in sources
- **Connection**: Linking disparate concepts
- **Practicality**: Actionable recommendations
- **Surprise**: Counter-intuitive insights

### Human-Likeness
- AI detection resistance
- Voice consistency
- Natural flow
- Specificity

## Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OLLAMA_BASE_URL` | Local Ollama server | `http://localhost:11434` |
| `CRAWL_DELAY` | Seconds between requests | `2.0` |
| `MAX_ARTICLES` | Maximum articles to crawl | `100` |
| `DEFAULT_MODEL` | Primary LLM model | `deepseek/deepseek-chat` |

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=mindforge_dankoe

# Run specific test file
poetry run pytest tests/test_style_analyzer.py
```

### Code Quality

```bash
# Format code
poetry run black mindforge_dankoe tests
poetry run isort mindforge_dankoe tests

# Lint
poetry run ruff check mindforge_dankoe

# Type check
poetry run mypy mindforge_dankoe
```

## Ethical Considerations

This project is intended for:
- Learning about writing styles and patterns
- Accelerating content creation workflow
- Developing your own authentic voice

Please use responsibly:
- Respect robots.txt and rate limits when crawling
- Don't misrepresent AI-generated content as human-written
- Use as a tool to enhance, not replace, human creativity

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Dan Koe for the inspiring writing style
- The open-source AI community for tools and libraries
