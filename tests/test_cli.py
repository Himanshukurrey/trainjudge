from click.testing import CliRunner

from trainjudge import __version__
from trainjudge.cli import main


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_subcommands_registered():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("diagnose", "audit", "train", "verify"):
        assert cmd in result.output
