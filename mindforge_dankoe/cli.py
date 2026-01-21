"""
MindForge CLI - Command line interface for the content generation system.

Provides commands for:
- Crawling Dan Koe articles
- Analyzing style patterns
- Generating content
- Evaluating outputs
- Interactive writing assistance
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mindforge_dankoe.config.settings import settings

app = typer.Typer(
    name="mindforge",
    help="MindForge AI - Dan Koe Style Content Generation System",
    add_completion=False,
)

console = Console()


@app.command()
def crawl(
    max_articles: int = typer.Option(50, "--max", help="Maximum articles to crawl"),
    delay: float = typer.Option(2.0, "--delay", help="Delay between requests in seconds"),
    output_dir: Optional[str] = typer.Option(None, "--output", help="Output directory"),
):
    """Crawl Dan Koe's newsletter articles."""
    from mindforge_dankoe.crawler.dankoe_spider import DanKoeSpider

    console.print(Panel("Starting Dan Koe Newsletter Crawler", style="bold blue"))

    output_path = Path(output_dir) if output_dir else None

    async def run_crawl():
        spider = DanKoeSpider(
            output_dir=output_path,
            max_articles=max_articles,
            crawl_delay=delay,
        )
        return await spider.crawl_all()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Crawling articles...", total=None)
        articles = asyncio.run(run_crawl())

    console.print(f"\n[green]Successfully crawled {len(articles)} articles[/green]")

    # Show summary table
    if articles:
        table = Table(title="Crawled Articles")
        table.add_column("Title", style="cyan")
        table.add_column("Words", justify="right")
        table.add_column("Date")

        for article in articles[:10]:
            table.add_row(
                article.title[:50] + "..." if len(article.title) > 50 else article.title,
                str(article.word_count),
                article.date_published or "Unknown",
            )

        if len(articles) > 10:
            table.add_row("...", "...", "...")

        console.print(table)


@app.command()
def process(
    input_dir: Optional[str] = typer.Option(None, "--input", help="Input directory with raw articles"),
    output_dir: Optional[str] = typer.Option(None, "--output", help="Output directory"),
):
    """Process crawled articles for analysis."""
    from mindforge_dankoe.crawler.article_processor import ArticleProcessor
    from mindforge_dankoe.crawler.dankoe_spider import DanKoeSpider

    console.print(Panel("Processing Crawled Articles", style="bold blue"))

    # Load cached articles
    input_path = Path(input_dir) if input_dir else None
    spider = DanKoeSpider(output_dir=input_path)
    crawled = spider.load_cached_articles()

    if not crawled:
        console.print("[yellow]No crawled articles found. Run 'crawl' first.[/yellow]")
        return

    console.print(f"Found {len(crawled)} articles to process")

    # Process articles
    output_path = Path(output_dir) if output_dir else None
    processor = ArticleProcessor(output_dir=output_path)
    processed = processor.process_all(crawled)

    console.print(f"\n[green]Successfully processed {len(processed)} articles[/green]")


@app.command()
def analyze(
    output_file: Optional[str] = typer.Option(None, "--output", help="Output file for analysis"),
):
    """Analyze articles for style and thinking patterns."""
    from mindforge_dankoe.analysis.style_analyzer import StyleAnalyzer
    from mindforge_dankoe.analysis.thinking_extractor import ThinkingExtractor
    from mindforge_dankoe.crawler.article_processor import ArticleProcessor

    console.print(Panel("Analyzing Dan Koe's Style & Thinking Patterns", style="bold blue"))

    # Load processed articles
    processor = ArticleProcessor()
    articles = processor.load_processed_articles()

    if not articles:
        console.print("[yellow]No processed articles found. Run 'process' first.[/yellow]")
        return

    console.print(f"Analyzing {len(articles)} articles...")

    # Create style fingerprint
    style_analyzer = StyleAnalyzer()
    fingerprint = style_analyzer.create_fingerprint(articles)
    style_analyzer.save_fingerprint()

    # Extract thinking patterns
    thinking_extractor = ThinkingExtractor()
    for article in articles[:10]:  # Analyze top 10
        thinking_extractor.analyze_article(article)
    thinking_extractor.save_patterns()

    # Show summary
    console.print("\n[bold]Style Fingerprint Summary:[/bold]")
    console.print(style_analyzer.get_style_summary())

    # Show thinking pattern summary
    synthesis = thinking_extractor.synthesize_patterns(thinking_extractor.extracted_analyses)
    if synthesis:
        console.print("\n[bold]Common Thinking Patterns:[/bold]")
        for pattern in synthesis.get("common_patterns", [])[:5]:
            console.print(f"  • {pattern}")

    console.print(f"\n[green]Analysis complete. Files saved to {settings.get_processed_data_path()}[/green]")


@app.command()
def generate(
    topic: str = typer.Argument(..., help="Topic to write about"),
    language: str = typer.Option("en", "--lang", help="Target language (en/zh)"),
    words: int = typer.Option(1500, "--words", help="Target word count"),
    output_file: Optional[str] = typer.Option(None, "--output", help="Output file"),
    research: bool = typer.Option(True, "--research/--no-research", help="Conduct web research"),
):
    """Generate an article on a topic in Dan Koe's style."""
    from mindforge_dankoe.analysis.style_analyzer import StyleAnalyzer
    from mindforge_dankoe.generation.cognitive_processor import CognitiveProcessor
    from mindforge_dankoe.generation.research_engine import ResearchEngine, ResearchResult
    from mindforge_dankoe.generation.style_generator import StyleGenerator

    console.print(Panel(f"Generating Article: {topic}", style="bold blue"))

    async def do_research():
        engine = ResearchEngine()
        try:
            return await engine.research_topic(topic)
        finally:
            await engine.close()

    # Research phase
    research_result = None
    if research:
        console.print("Conducting web research...")
        try:
            research_result = asyncio.run(do_research())
            console.print(f"  Found {len(research_result.sources)} sources")
            console.print(f"  Extracted {len(research_result.key_findings)} key findings")
        except Exception as e:
            console.print(f"[yellow]Research unavailable: {e}[/yellow]")
            research_result = ResearchResult(topic=topic)

    if not research_result:
        research_result = ResearchResult(topic=topic)

    # Cognitive processing
    console.print("Processing through thinking framework...")
    processor = CognitiveProcessor()
    cognitive_output = processor.apply_thinking_template(topic, research_result)
    console.print(f"  Generated {len(cognitive_output.insights)} insights")
    console.print(f"  Creativity score: {cognitive_output.creativity_scores.get('overall_score', 'N/A')}")

    # Style generation
    console.print("Generating article...")
    style_analyzer = StyleAnalyzer()
    try:
        fingerprint = style_analyzer.load_fingerprint()
    except FileNotFoundError:
        fingerprint = None
        console.print("[yellow]No style fingerprint found. Using defaults.[/yellow]")

    generator = StyleGenerator(style_fingerprint=fingerprint)
    article = generator.generate_article(
        cognitive_output,
        target_language=language,
        word_count_target=words,
    )

    # Save article
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = generator.save_article(article)

    console.print(f"\n[green]Article generated successfully![/green]")
    console.print(f"Title: {article.title}")
    console.print(f"Words: {article.word_count}")
    console.print(f"Saved to: {output_path}")

    # Show preview
    console.print("\n[bold]Preview:[/bold]")
    preview = article.full_content[:500] + "..." if len(article.full_content) > 500 else article.full_content
    console.print(Panel(preview, title="Article Preview"))


@app.command()
def evaluate(
    article_path: str = typer.Argument(..., help="Path to article JSON file"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis"),
):
    """Evaluate a generated article for quality and similarity."""
    from mindforge_dankoe.evaluation.creativity_scorer import CreativityScorer
    from mindforge_dankoe.evaluation.human_likeness_tests import HumanLikenessEvaluator
    from mindforge_dankoe.evaluation.similarity_metrics import SimilarityMetrics
    from mindforge_dankoe.generation.style_generator import GeneratedArticle

    console.print(Panel("Evaluating Article", style="bold blue"))

    # Load article
    with open(article_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    article = GeneratedArticle(
        title=data.get("title", ""),
        full_content=data.get("full_content", ""),
        word_count=data.get("word_count", 0),
    )

    # Run evaluations
    console.print(f"Evaluating: {article.title}\n")

    # Similarity
    console.print("[bold]Style Similarity:[/bold]")
    similarity_eval = SimilarityMetrics()
    similarity = similarity_eval.calculate_similarity(article)
    console.print(f"  Overall: {similarity.overall_score:.1%}")
    console.print(f"  {similarity.interpretation}")

    # Creativity
    console.print("\n[bold]Creativity:[/bold]")
    creativity_eval = CreativityScorer()
    creativity = creativity_eval.score_article(article)
    console.print(f"  Overall: {creativity.overall_score:.2f}")
    console.print(f"  {creativity.interpretation}")

    # Human-likeness
    console.print("\n[bold]Human-Likeness:[/bold]")
    human_eval = HumanLikenessEvaluator()
    human = human_eval.evaluate(article)
    console.print(f"  Overall: {human.overall_score:.1%}")
    console.print(f"  {human.interpretation}")

    if detailed:
        # Show detailed breakdown
        console.print("\n[bold]Detailed Breakdown:[/bold]")

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Score")

        table.add_row("Thinking Pattern Match", f"{similarity.thinking_pattern_similarity:.1%}")
        table.add_row("Stylistic Match", f"{similarity.stylistic_similarity:.1%}")
        table.add_row("Thematic Alignment", f"{similarity.thematic_alignment:.1%}")
        table.add_row("Tone Consistency", f"{similarity.tone_consistency:.1%}")
        table.add_row("", "")
        table.add_row("Novelty", f"{creativity.novelty.score:.2f}")
        table.add_row("Connection", f"{creativity.connection.score:.2f}")
        table.add_row("Practicality", f"{creativity.practicality.score:.2f}")
        table.add_row("Surprise", f"{creativity.surprise.score:.2f}")
        table.add_row("", "")
        table.add_row("AI Detection Risk", f"{human.ai_detection_risk:.1%}")
        table.add_row("Voice Consistency", f"{human.voice_consistency:.1%}")
        table.add_row("Natural Flow", f"{human.natural_flow:.1%}")

        console.print(table)

        # Show recommendations
        if human.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in human.recommendations[:5]:
                console.print(f"  • {rec}")


@app.command()
def interactive():
    """Start an interactive writing session."""
    from mindforge_dankoe.interactive import InteractiveWritingAssistant

    console.print(Panel("Interactive Writing Assistant", style="bold green"))
    console.print("Type 'help' for commands, 'quit' to exit.\n")

    assistant = InteractiveWritingAssistant()
    assistant.run()


@app.command()
def status():
    """Show current system status and configuration."""
    console.print(Panel("MindForge System Status", style="bold blue"))

    # Check directories
    data_dir = settings.data_dir
    output_dir = settings.output_dir

    table = Table(title="Directory Status")
    table.add_column("Directory", style="cyan")
    table.add_column("Path")
    table.add_column("Status")

    table.add_row("Data", str(data_dir), "✓ Exists" if data_dir.exists() else "✗ Missing")
    table.add_row("Output", str(output_dir), "✓ Exists" if output_dir.exists() else "✗ Missing")

    console.print(table)

    # Check API keys
    table2 = Table(title="API Configuration")
    table2.add_column("Service", style="cyan")
    table2.add_column("Status")

    table2.add_row("DeepSeek", "✓ Configured" if settings.deepseek_api_key else "✗ Not set")
    table2.add_row("OpenAI", "✓ Configured" if settings.openai_api_key else "✗ Not set")
    table2.add_row("Anthropic", "✓ Configured" if settings.anthropic_api_key else "✗ Not set")
    table2.add_row("Serper", "✓ Configured" if settings.serper_api_key else "✗ Not set")
    table2.add_row("Tavily", "✓ Configured" if settings.tavily_api_key else "✗ Not set")

    console.print(table2)

    # Check cached data
    raw_path = settings.get_raw_data_path() / "dankoe"
    processed_path = settings.get_processed_data_path() / "dankoe"

    raw_count = len(list(raw_path.glob("*.json"))) if raw_path.exists() else 0
    processed_count = len(list(processed_path.glob("*.json"))) if processed_path.exists() else 0

    console.print(f"\nCached Articles: {raw_count} raw, {processed_count} processed")

    # Check fingerprint
    fingerprint_path = settings.get_processed_data_path() / "style_fingerprint.json"
    console.print(f"Style Fingerprint: {'✓ Available' if fingerprint_path.exists() else '✗ Not created'}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
