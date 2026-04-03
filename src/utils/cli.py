from typing import Callable
from src.const import CONSOLE, LOGGER
from rich.panel import Panel


def cli_menu(question: str, options: dict[str, Callable]) -> None:
    """Display a CLI menu for user to select assignments and functionalities.

    Args:
        question (str): The question to be displayed.
        options (dict[str, Callable]): The options for the menu.
    """

    panel = Panel(f"[bold cyan]{question}[/bold cyan]")

    LOGGER.log_and_print(panel)

    # Create options display
    options_text = "\n".join(
        f"{i}. {option}" for i, option in enumerate(options.keys(), 1)
    )
    options_panel = Panel(options_text)

    LOGGER.log_and_print(options_panel)

    choice = CONSOLE.input("\n[bold cyan]Enter your choice:[/bold cyan] ").strip()

    LOGGER.debug(f"User selected option: {choice}")

    if choice.isdigit() and 1 <= int(choice) <= len(options):
        selected_option = list(options.values())[int(choice) - 1]
        selected_option()

    else:
        panel = Panel("[bold red]Invalid choice. Please try again.[/bold red]")
        LOGGER.log_and_print(panel)
