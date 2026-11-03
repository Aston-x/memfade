# Contributing to memfade

Thanks for your interest in contributing! This document outlines how to get started and the conventions we follow.

## Development Setup

```bash
git clone https://github.com/Aston-x/memfade.git
cd memfade
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

# No third-party dependencies needed for the core!
python -m unittest discover -s tests -t .
```

## Running Tests

```bash
# All tests
python -m unittest discover -s tests -t . -v

# Specific test file
python -m unittest tests.test_decay

# Koishi adapter tests (requires Node.js)
cd adapters/koishi
npm install
npm test
```

## Project Structure

- `memfade/` - Core engine (zero third-party dependencies)
- `adapters/` - Host-specific adapters (MaiBot, LangBot, AstrBot, NoneBot2, Koishi, HTTP)
- `scripts/` - Maintenance and diagnostic tools
- `tests/` - Test suite (318 Python tests + 6 Koishi tests)
- `docs/` - Architecture, decay algorithm, comparisons

## Coding Conventions

1. **Chinese comments and docstrings** - The project is primarily in Chinese
2. **No third-party dependencies in core** - Keep `memfade/` dependency-free
3. **Idempotent operations** - Decay calculations must be idempotent (recalculate from anchor, don't accumulate)
4. **Write tests** - Every new feature needs tests
5. **Document decisions** - Explain "why not the alternative" in docstrings

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass
5. Commit with clear messages (Chinese or English both fine)
6. Push to your fork and open a Pull Request

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Your environment (Python version, host adapter, OS)

## Feature Requests

Open an issue describing:
- The problem you're trying to solve
- Why existing features don't cover it
- Your proposed solution (if you have one)

## Questions?

Open a discussion or issue. We're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
