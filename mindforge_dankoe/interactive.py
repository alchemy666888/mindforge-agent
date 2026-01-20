"""
Interactive Writing Assistant - Real-time collaborative writing with style evolution tracking.

Features:
- Interactive topic selection and generation
- Real-time voice adjustment
- Style evolution tracking
- Personal knowledge integration
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.generation.style_generator import GeneratedArticle, StyleGenerator, VoiceSettings
from mindforge_dankoe.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class EditHistory:
    """Track user edits to learn preferences."""

    original_text: str
    edited_text: str
    edit_type: str  # "shorten", "lengthen", "rephrase", "delete", "add"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserStyleProfile:
    """User's emerging style preferences."""

    directness_preference: float = 0.5
    data_density_preference: float = 0.5
    practicality_preference: float = 0.5
    optimism_preference: float = 0.5
    formality_preference: float = 0.5

    edit_history: List[EditHistory] = field(default_factory=list)
    favorite_phrases: List[str] = field(default_factory=list)
    avoided_phrases: List[str] = field(default_factory=list)
    articles_generated: int = 0
    total_edits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "directness_preference": self.directness_preference,
            "data_density_preference": self.data_density_preference,
            "practicality_preference": self.practicality_preference,
            "optimism_preference": self.optimism_preference,
            "formality_preference": self.formality_preference,
            "favorite_phrases": self.favorite_phrases,
            "avoided_phrases": self.avoided_phrases,
            "articles_generated": self.articles_generated,
            "total_edits": self.total_edits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserStyleProfile":
        """Create from dictionary."""
        return cls(
            directness_preference=data.get("directness_preference", 0.5),
            data_density_preference=data.get("data_density_preference", 0.5),
            practicality_preference=data.get("practicality_preference", 0.5),
            optimism_preference=data.get("optimism_preference", 0.5),
            formality_preference=data.get("formality_preference", 0.5),
            favorite_phrases=data.get("favorite_phrases", []),
            avoided_phrases=data.get("avoided_phrases", []),
            articles_generated=data.get("articles_generated", 0),
            total_edits=data.get("total_edits", 0),
        )


class InteractiveWritingAssistant:
    """
    Interactive assistant for collaborative writing.

    Features:
    - Step-by-step article generation
    - Real-time adjustments
    - Style preference learning
    - Personal voice development
    """

    def __init__(self, profile_path: Optional[Path] = None):
        """
        Initialize the assistant.

        Args:
            profile_path: Path to user profile file
        """
        self.profile_path = profile_path or settings.get_data_path() / "user_profile.json"
        self.profile = self._load_profile()
        self.current_article: Optional[GeneratedArticle] = None
        self.generator: Optional[StyleGenerator] = None
        self._running = False

    def _load_profile(self) -> UserStyleProfile:
        """Load user profile from disk."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r") as f:
                    data = json.load(f)
                return UserStyleProfile.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load profile: {e}")

        return UserStyleProfile()

    def _save_profile(self):
        """Save user profile to disk."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.profile_path, "w") as f:
            json.dump(self.profile.to_dict(), f, indent=2)

    def run(self):
        """Run the interactive session."""
        self._running = True

        while self._running:
            try:
                command = Prompt.ask("\n[bold cyan]mindforge[/bold cyan]", default="help")
                self._handle_command(command.strip().lower())
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def _handle_command(self, command: str):
        """Handle user command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            "help": self._cmd_help,
            "new": self._cmd_new,
            "generate": self._cmd_generate,
            "adjust": self._cmd_adjust,
            "show": self._cmd_show,
            "save": self._cmd_save,
            "profile": self._cmd_profile,
            "voice": self._cmd_voice,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }

        handler = commands.get(cmd, self._cmd_unknown)
        handler(args)

    def _cmd_help(self, args: str):
        """Show help information."""
        console.print(Panel(
            """[bold]Available Commands:[/bold]

[cyan]new[/cyan] <topic>     - Start a new article on a topic
[cyan]generate[/cyan]        - Generate content for current article
[cyan]adjust[/cyan] <param>  - Adjust generation parameters
[cyan]show[/cyan]            - Show current article
[cyan]save[/cyan] [filename] - Save current article
[cyan]profile[/cyan]         - Show/edit your style profile
[cyan]voice[/cyan]           - Adjust voice settings
[cyan]help[/cyan]            - Show this help
[cyan]quit[/cyan]            - Exit the assistant

[bold]Examples:[/bold]
  new AI agents and the future of work
  adjust directness 0.9
  voice --more-direct
""",
            title="Help",
            style="blue",
        ))

    def _cmd_new(self, args: str):
        """Start a new article."""
        if not args:
            topic = Prompt.ask("What topic would you like to write about?")
        else:
            topic = args

        console.print(f"\n[green]Starting new article on: {topic}[/green]")

        # Initialize generator with user's voice preferences
        voice = VoiceSettings(
            directness=self.profile.directness_preference,
            data_density=self.profile.data_density_preference,
            practicality_bias=self.profile.practicality_preference,
            optimism_level=self.profile.optimism_preference,
            formality=self.profile.formality_preference,
        )

        self.generator = StyleGenerator(voice_settings=voice)

        console.print("\nReady to generate. Use [cyan]generate[/cyan] to create content.")
        console.print("Use [cyan]adjust[/cyan] to modify parameters first if needed.")

        # Store topic for generation
        self._current_topic = topic

    def _cmd_generate(self, args: str):
        """Generate article content."""
        if not hasattr(self, "_current_topic"):
            console.print("[yellow]No topic set. Use 'new <topic>' first.[/yellow]")
            return

        console.print("\n[bold]Generating article...[/bold]")

        # Import here to avoid circular imports
        from mindforge_dankoe.generation.cognitive_processor import CognitiveProcessor
        from mindforge_dankoe.generation.research_engine import ResearchResult

        # For now, create a simple research result
        research = ResearchResult(topic=self._current_topic)

        # Generate cognitive output
        processor = CognitiveProcessor()
        cognitive_output = processor.apply_thinking_template(self._current_topic, research)

        # Generate article
        self.current_article = self.generator.generate_article(
            cognitive_output,
            target_language="en",
            word_count_target=1500,
        )

        # Update profile
        self.profile.articles_generated += 1
        self._save_profile()

        console.print(f"\n[green]Generated: {self.current_article.title}[/green]")
        console.print(f"Word count: {self.current_article.word_count}")

        # Show preview
        self._cmd_show("")

    def _cmd_adjust(self, args: str):
        """Adjust generation parameters."""
        if not args:
            # Show current settings
            if self.generator:
                voice = self.generator.voice
                console.print("\n[bold]Current Voice Settings:[/bold]")
                console.print(f"  directness:    {voice.directness:.1f}")
                console.print(f"  data_density:  {voice.data_density:.1f}")
                console.print(f"  practicality:  {voice.practicality_bias:.1f}")
                console.print(f"  optimism:      {voice.optimism_level:.1f}")
                console.print(f"  formality:     {voice.formality:.1f}")
                console.print("\nUse: adjust <param> <value> (0.0-1.0)")
            return

        parts = args.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: adjust <param> <value>[/yellow]")
            return

        param = parts[0]
        try:
            value = float(parts[1])
            value = max(0.0, min(1.0, value))
        except ValueError:
            console.print("[red]Value must be a number between 0.0 and 1.0[/red]")
            return

        if not self.generator:
            self.generator = StyleGenerator()

        param_map = {
            "directness": "directness",
            "data": "data_density",
            "data_density": "data_density",
            "practicality": "practicality_bias",
            "practical": "practicality_bias",
            "optimism": "optimism_level",
            "formality": "formality",
            "formal": "formality",
        }

        if param.lower() not in param_map:
            console.print(f"[yellow]Unknown parameter: {param}[/yellow]")
            console.print(f"Available: {', '.join(param_map.keys())}")
            return

        attr = param_map[param.lower()]
        setattr(self.generator.voice, attr, value)
        console.print(f"[green]Set {attr} to {value:.1f}[/green]")

        # Update profile preference
        profile_map = {
            "directness": "directness_preference",
            "data_density": "data_density_preference",
            "practicality_bias": "practicality_preference",
            "optimism_level": "optimism_preference",
            "formality": "formality_preference",
        }

        if attr in profile_map:
            setattr(self.profile, profile_map[attr], value)
            self._save_profile()

    def _cmd_show(self, args: str):
        """Show current article."""
        if not self.current_article:
            console.print("[yellow]No article generated yet.[/yellow]")
            return

        console.print(Panel(
            f"[bold]{self.current_article.title}[/bold]\n\n"
            + self.current_article.full_content[:2000]
            + ("..." if len(self.current_article.full_content) > 2000 else ""),
            title="Current Article",
        ))

    def _cmd_save(self, args: str):
        """Save current article."""
        if not self.current_article:
            console.print("[yellow]No article to save.[/yellow]")
            return

        if args:
            filepath = Path(args)
        else:
            filepath = self.generator.save_article(self.current_article)

        console.print(f"[green]Saved to: {filepath}[/green]")

    def _cmd_profile(self, args: str):
        """Show or edit user profile."""
        table = Table(title="Your Style Profile")
        table.add_column("Preference", style="cyan")
        table.add_column("Value")

        table.add_row("Directness", f"{self.profile.directness_preference:.1f}")
        table.add_row("Data Density", f"{self.profile.data_density_preference:.1f}")
        table.add_row("Practicality", f"{self.profile.practicality_preference:.1f}")
        table.add_row("Optimism", f"{self.profile.optimism_preference:.1f}")
        table.add_row("Formality", f"{self.profile.formality_preference:.1f}")
        table.add_row("", "")
        table.add_row("Articles Generated", str(self.profile.articles_generated))
        table.add_row("Total Edits", str(self.profile.total_edits))

        console.print(table)

        if self.profile.favorite_phrases:
            console.print("\n[bold]Favorite Phrases:[/bold]")
            for phrase in self.profile.favorite_phrases[:5]:
                console.print(f"  • {phrase}")

    def _cmd_voice(self, args: str):
        """Adjust voice with natural language."""
        if not args:
            console.print("""
[bold]Voice Adjustment:[/bold]
  voice --more-direct     Increase directness
  voice --more-casual     Decrease formality
  voice --more-practical  Increase actionability
  voice --more-optimistic Increase optimism
  voice --reset           Reset to Dan Koe defaults
""")
            return

        if not self.generator:
            self.generator = StyleGenerator()

        adjustments = {
            "--more-direct": ("directness", 0.1),
            "--less-direct": ("directness", -0.1),
            "--more-casual": ("formality", -0.1),
            "--more-formal": ("formality", 0.1),
            "--more-practical": ("practicality_bias", 0.1),
            "--more-theoretical": ("practicality_bias", -0.1),
            "--more-optimistic": ("optimism_level", 0.1),
            "--more-skeptical": ("optimism_level", -0.1),
            "--more-data": ("data_density", 0.1),
            "--less-data": ("data_density", -0.1),
        }

        if args == "--reset":
            self.generator.voice = VoiceSettings.dankoe_default()
            console.print("[green]Voice reset to Dan Koe defaults[/green]")
            return

        if args not in adjustments:
            console.print(f"[yellow]Unknown adjustment: {args}[/yellow]")
            return

        attr, delta = adjustments[args]
        current = getattr(self.generator.voice, attr)
        new_value = max(0.0, min(1.0, current + delta))
        setattr(self.generator.voice, attr, new_value)

        console.print(f"[green]Adjusted {attr}: {current:.1f} → {new_value:.1f}[/green]")

    def _cmd_quit(self, args: str):
        """Exit the assistant."""
        if Confirm.ask("Save profile before exiting?"):
            self._save_profile()
        self._running = False
        console.print("[green]Goodbye![/green]")

    def _cmd_unknown(self, args: str):
        """Handle unknown commands."""
        console.print("[yellow]Unknown command. Type 'help' for available commands.[/yellow]")
