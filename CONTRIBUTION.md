# Contributing to EvoAgentX

Thank you for considering contributing to **EvoAgentX** – an automatic agentic workflow generation and evolving framework! We appreciate your interest and contributions to improving the project.

## 🚀 Getting Started

1. **Fork the Repository**: Click the 'Fork' button on the top right of this repo to create your own copy.
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/your-username/EvoAgentX.git
   ```
3. **Navigate to the Project Directory**:
   ```bash
   cd EvoAgentX
   ```
4. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
5. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 How to Contribute

### **1. Reporting Bugs** 🐞
- Ensure the bug has not been reported in the [Issues](https://github.com/EvoAgentX/issues) tab.
- Provide detailed steps to reproduce the issue.
- Include logs, screenshots, and environment details.
- Use a clear title and concise description.

### **2. Suggesting Features** 💡
- Check if a similar feature request exists in [Issues](https://github.com/EvoAgentX/issues).
- Provide a use case explaining why this feature is beneficial.
- Suggest a possible implementation approach if possible.

### **3. Code Contributions** 👨‍💻
#### **Step 1: Create a Branch**
Before making changes, create a new branch:
```bash
git checkout -b feature/your-feature-name
```

#### **Step 2: Implement Your Changes**
- Follow the project’s coding style.
- Write clean, maintainable, and well-documented code.
- Add relevant unit tests for your changes.

#### **Step 3: Commit Your Changes**
- Follow conventional commit messages:
  ```bash
  git commit -m "feat: add new agent workflow module"
  ```
- Use descriptive commit messages.

#### **Step 4: Push Your Changes**
```bash
git push origin feature/your-feature-name
```

#### **Step 5: Submit a Pull Request**
- Navigate to the main repository and create a **Pull Request** (PR).
- Provide a clear description of your changes.
- Reference the related issue if applicable.
- Wait for a review and respond to feedback promptly.

## 📏 Coding Guidelines

- Use **Python 3.8+**.
- Follow **PEP8** style guide.
- Use **docstrings** for functions and classes.
- Run **pre-commit hooks** before pushing:
  ```bash
  pip install pre-commit
  pre-commit install
  pre-commit run --all-files
  ```

## 🧪 Testing
Before submitting a PR, ensure that all tests pass:
```bash
pytest tests/
```

## 🔄 Syncing Your Fork
To stay updated with the latest changes:
```bash
git fetch upstream
```
```bash
git merge upstream/main
```

## 🤝 Community Guidelines
- Be respectful and inclusive.
- Provide constructive feedback.
- Keep discussions relevant to the project.

## 📩 Contact
For any questions, feel free to open an issue or reach out to the maintainers.

Happy coding! 🚀
