# StatementScreen

Takes a bank-statement PDF and returns a verified income profile, a tamper verdict,
and a plain-language reason for every finding, using classical ML and pattern
recognition (no LLMs).

> **All data in this project is synthetic.** No real bank statements, customer data
> or real bank branding are used anywhere.

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # create .venv and install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format
```

## License

MIT. See [LICENSE](LICENSE).
