# Contributing to DocuFind

First off, thank you for considering contributing to DocuFind! It's people like you that make DocuFind such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by the DocuFind Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots if possible

### Suggesting Enhancements

If you have a suggestion for the project, we'd love to hear it. Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed enhancement
* Examples of how the enhancement would be used
* If possible, mock-ups or examples of similar features in other projects

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python styleguide
* Include thoughtfully-worded, well-structured tests
* Document new code
* End all files with a newline

## Development Process

1. Fork the repo
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Setting Up Development Environment

1. Install Python 3.7+
2. Clone your fork
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the development server:
   ```bash
   python main.py
   ```

## Style Guide

* Follow PEP 8
* Use meaningful variable names
* Write docstrings for all public methods
* Keep functions focused and small
* Comment complex logic

## Testing

* Write unit tests for new features
* Ensure all tests pass before submitting PR
* Include both positive and negative test cases
* Test edge cases

## Documentation

* Update the README.md if needed
* Document all new features
* Include docstrings in your code
* Comment non-obvious code sections

## Questions?

Feel free to open an issue with your question or contact the maintainers directly.

Thank you for contributing to DocuFind! 🎉