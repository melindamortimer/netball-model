from click.testing import CliRunner

from netball_model.cli import main


def test_cli_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Netball betting model CLI" in result.output
