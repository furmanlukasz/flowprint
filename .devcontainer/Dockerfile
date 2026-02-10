FROM mcr.microsoft.com/devcontainers/python:3.11

# Install UV for vscode user
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH for all future commands
ENV PATH="/home/vscode/.local/bin:${PATH}"
