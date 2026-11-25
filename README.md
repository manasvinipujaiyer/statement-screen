# StatementScreen

Takes an Indian bank-statement PDF and returns a verified income profile, a tamper
verdict, and a plain-language reason for every finding, using classical ML and pattern
recognition (no LLMs).

> **All data in this repo is synthetic.** Real statements are used only for private
> evaluation, stay on my laptop, and are never published.
>
> The banks, IFSC-style codes, UPI handles, employers and merchants in the synthetic data
> are fictional, and every generated PDF carries a "SYNTHETIC DOCUMENT" footer.

## Development setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # create .venv and install dependencies
uv run pytest        # run tests
uv run ruff check .  # lint
uv run ruff format . # format
```

## Documentation

The detailed write-ups, one per stage:

- [Generating synthetic statements](docs/data_generation.md)
- [Parsing](docs/parsing.md)
- [Pattern detection](docs/patterns.md)
- [Classifier vs rules](docs/classifier_vs_rules.md)
- [Income profile](docs/income_profile.md)
- [Tamper detection](docs/tamper_detection.md)
- [The service](docs/service.md)
- [The blind tamper test](docs/blind_test.md)
- [Evaluation protocol](docs/evaluation_protocol.md)

## License

MIT. See [LICENSE](LICENSE).
