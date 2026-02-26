from __future__ import annotations

from rich.console import Console
from rich.table import Table


def display_predictions(predictions: list[dict]):
    """Render predictions as a rich table in the terminal."""
    console = Console()
    table = Table(title="Match Predictions", show_lines=True)

    table.add_column("Match", style="bold")
    table.add_column("Pred Margin", justify="right")
    table.add_column("Pred Total", justify="right")
    table.add_column("Model Win%", justify="right")
    table.add_column("Betfair Odds", justify="right")
    table.add_column("Implied%", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Value?", justify="center")

    for p in predictions:
        edge_str = f"{p.get('edge', 0):.1%}"
        value_str = "YES" if p.get("is_value") else "-"
        value_style = "bold green" if p.get("is_value") else "dim"

        odds_str = f"{p.get('odds', '-')}" if p.get("odds") else "-"
        implied_str = f"{p.get('implied_prob', 0):.1%}" if p.get("implied_prob") else "-"

        table.add_row(
            f"{p['home_team']} v {p['away_team']}",
            f"{p['predicted_margin']:+.1f}",
            f"{p['predicted_total']:.0f}",
            f"{p['win_probability']:.1%}",
            odds_str,
            implied_str,
            edge_str,
            f"[{value_style}]{value_str}[/{value_style}]",
        )

    console.print(table)
