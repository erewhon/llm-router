"""Terminal dashboard for LLM Router monitoring.

Polls the dashboard API and displays node metrics + model health
in a Rich live-updating TUI.

Run:  llm-router-tui
Or:   uv run python -m llm_router.tui
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

import click
import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.style import Style


def _key_pressed() -> str | None:
    """Non-blocking check for a keypress. Returns the key or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _bar(pct: float, width: int = 20) -> Text:
    """Render a colored progress bar."""
    if pct >= 90:
        color = "red"
    elif pct >= 70:
        color = "yellow"
    else:
        color = "green"
    filled = int(pct / 100 * width)
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * (width - filled), style="dim")
    bar.append(f" {pct:.0f}%", style="bold " + color)
    return bar


def _health_dot(state: str) -> Text:
    """Render a colored health indicator."""
    colors = {
        "running": "green",
        "healthy": "green",
        "routed": "green",
        "starting": "yellow",
        "stopped": "dim",
        "disabled": "dim",
        "error": "red",
        "unhealthy": "red",
        "unknown": "dim",
    }
    color = colors.get(state, "dim")
    t = Text()
    t.append("● ", style=color)
    t.append(state, style=color)
    return t


def _fetch_data(base_url: str) -> dict | None:
    """Fetch dashboard data."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base_url}/api/models")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def _fetch_metrics(base_url: str) -> dict | None:
    """Fetch node metrics."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base_url}/api/node-metrics")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def _build_node_panel(name: str, node: dict, metrics: dict) -> Panel:
    """Build a Rich panel for a single node."""
    m = metrics.get(name, {})
    reachable = m.get("reachable", False)

    lines = Text()
    lines.append(f"Host: ", style="dim")
    lines.append(f"{node['host']}\n")
    lines.append(f"GPU:  ", style="dim")
    lines.append(f"{node['gpu'].upper()} · {node['vram_gb']} GB\n")

    if reachable and m.get("vram_pct") is not None:
        lines.append("MEM   ", style="dim")
        lines.append_text(_bar(m["vram_pct"]))
        lines.append(f"  {m.get('vram_used_gb', '?')}/{m.get('vram_total_gb', '?')} GB\n")

        busy = m.get("gpu_busy_pct")
        if busy is not None:
            lines.append("GPU   ", style="dim")
            lines.append_text(_bar(busy))
            lines.append("\n")

        if m.get("disk_free_gb") is not None and m.get("disk_total_gb") is not None:
            disk_pct = (m["disk_total_gb"] - m["disk_free_gb"]) / m["disk_total_gb"] * 100
            lines.append("DISK  ", style="dim")
            lines.append_text(_bar(disk_pct))
            lines.append(f"  {m['disk_free_gb']:.0f} GB free\n")
    elif reachable:
        lines.append("No GPU metrics\n", style="dim italic")
    else:
        lines.append("Agent offline\n", style="red italic")

    # Services
    for svc in m.get("services", []):
        dot = "green" if svc.get("reachable") else "dim"
        lines.append("● ", style=dot)
        lines.append(f"{svc.get('label', svc.get('name', '?'))}\n", style="bold" if svc.get("reachable") else "dim")

    border_style = "green" if reachable else "red"
    return Panel(lines, title=f"[bold]{name}[/bold]", border_style=border_style, expand=True)


def _build_model_table(models: list[dict], show_disabled: bool = False) -> Table:
    """Build the models table."""
    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Backend", no_wrap=True)
    table.add_column("Node(s)", no_wrap=True)
    table.add_column("VRAM", justify="right", no_wrap=True)
    table.add_column("Aliases", no_wrap=True)
    table.add_column("Tags", no_wrap=True)
    table.add_column("Health", no_wrap=True)

    for m in models:
        if m.get("enabled") is False and not show_disabled:
            continue

        disabled = m.get("enabled") is False
        dim = "dim" if disabled else ""

        # Node string
        nodes = m.get("nodes", [])
        head = m.get("head_node")
        if nodes:
            node_parts = []
            for n in nodes:
                if n == head and len(nodes) > 1:
                    node_parts.append(f"[bold]{n}[/bold]")
                else:
                    node_parts.append(n)
            node_str = ", ".join(node_parts)
        else:
            node_str = "[dim]cloud[/dim]"

        # Aliases
        aliases = ", ".join(m.get("aliases", []))

        # Tags
        tags = " ".join(m.get("tags", []))

        # VRAM
        vram = f"{m['vram_gb']} GB" if m.get("vram_gb") else "—"

        # Health
        agent_state = m.get("agent_state")
        health = m.get("health", "unknown")
        if agent_state:
            health_text = _health_dot(agent_state)
        else:
            display = "healthy" if health == "routed" else health
            health_text = _health_dot(display)

        style = Style(dim=disabled)
        table.add_row(
            m["id"],
            m.get("backend", ""),
            node_str,
            vram,
            aliases or "—",
            tags,
            health_text,
            style=style,
        )

    return table


def _build_display(data: dict, metrics: dict | None, show_disabled: bool) -> Table:
    """Build the full display layout."""
    layout = Table.grid(expand=True)
    layout.add_row()

    # Stats bar
    models = data.get("models", [])
    enabled = [m for m in models if m.get("enabled") is not False]
    healthy = [m for m in enabled if m.get("health") in ("healthy", "routed") or m.get("agent_state") == "running"]

    stats = Text()
    stats.append(f"  Models: ", style="dim")
    stats.append(f"{len(enabled)}", style="bold cyan")
    stats.append(f"  Nodes: ", style="dim")
    stats.append(f"{data.get('node_count', 0)}", style="bold cyan")
    stats.append(f"  Healthy: ", style="dim")
    stats.append(f"{len(healthy)}", style="bold green")
    if show_disabled:
        disabled_count = len(models) - len(enabled)
        stats.append(f"  Disabled: ", style="dim")
        stats.append(f"{disabled_count}", style="dim")
    layout.add_row(stats)
    layout.add_row(Text())

    # Node panels
    nm = metrics or data.get("node_metrics", {})
    nodes = data.get("nodes", {})
    if nodes:
        panels = [_build_node_panel(name, node, nm) for name, node in nodes.items()]
        layout.add_row(Columns(panels, equal=True, expand=True))
        layout.add_row(Text())

    # Models table
    layout.add_row(Panel(
        _build_model_table(models, show_disabled),
        title="[bold]Models[/bold]",
        border_style="blue",
        expand=True,
    ))

    return layout


def _render_frame(data: dict, metrics: dict | None, show_disabled: bool, interval: int) -> Table:
    """Render a single frame of the TUI."""
    title = Text()
    title.append(" LLM Router ", style="bold white on blue")
    title.append(f"  {time.strftime('%H:%M:%S')}  ", style="dim")
    title.append(f"refresh: {interval}s  ", style="dim")
    title.append("q", style="bold")
    title.append(" quit", style="dim")

    display = Table.grid(expand=True)
    display.add_row(title)
    display.add_row(Text())
    display.add_row(_build_display(data, metrics, show_disabled))
    return display


@click.command()
@click.option("--url", default="http://localhost:4011", help="Dashboard API URL")
@click.option("--interval", "-n", default=2, type=int, help="Refresh interval in seconds")
@click.option("--show-disabled", "-d", is_flag=True, help="Show disabled models")
def cli(url: str, interval: int, show_disabled: bool) -> None:
    """LLM Router terminal dashboard."""
    console = Console()

    # Initial fetch
    data = _fetch_data(url)
    if not data:
        console.print(f"[red]Could not connect to dashboard at {url}[/red]")
        console.print("Make sure the dashboard is running and accessible.")
        raise SystemExit(1)

    # Set terminal to raw mode for key detection
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        tick = 0
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            # Render immediately with initial data
            live.update(_render_frame(data, None, show_disabled, interval))

            while True:
                try:
                    # Check for keypress
                    key = _key_pressed()
                    if key and key.lower() == "q":
                        break

                    # Full refresh every 30s, metrics-only otherwise
                    if tick % max(1, 30 // interval) == 0:
                        fresh = _fetch_data(url)
                        if fresh:
                            data = fresh

                    metrics = _fetch_metrics(url)
                    live.update(_render_frame(data, metrics, show_disabled, interval))
                    time.sleep(interval)
                    tick += 1

                except KeyboardInterrupt:
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    cli()
