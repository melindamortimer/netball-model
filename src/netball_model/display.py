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
    table.add_column("Market", justify="center")
    table.add_column("Side", justify="center")
    table.add_column("Edge", justify="right")
    table.add_column("Odds", justify="right")

    for p in predictions:
        match_label = f"{p['home_team']} v {p['away_team']}"
        value_bets = p.get("value_bets", [])

        if not value_bets:
            table.add_row(
                match_label,
                f"{p['predicted_margin']:+.1f}",
                f"{p['predicted_total']:.0f}",
                f"{p['win_probability']:.1%}",
                "-", "-", "-", "-",
            )
        else:
            for idx, vb in enumerate(value_bets):
                edge = vb.get("edge", 0)
                is_value = edge >= 0.05
                edge_style = "bold green" if is_value else "dim"
                table.add_row(
                    match_label if idx == 0 else "",
                    f"{p['predicted_margin']:+.1f}" if idx == 0 else "",
                    f"{p['predicted_total']:.0f}" if idx == 0 else "",
                    f"{p['win_probability']:.1%}" if idx == 0 else "",
                    vb.get("market", "-"),
                    vb.get("side", "-"),
                    f"[{edge_style}]{edge:+.1%}[/{edge_style}]",
                    f"{vb.get('odds', '-')}" if vb.get("odds") else "-",
                )

    console.print(table)
