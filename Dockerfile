FROM python:3.11-slim

RUN useradd -m -u 10001 cono_sandbox
WORKDIR /app

COPY pyproject.toml CLAUDE.MD ./
COPY packages ./packages
COPY scenarios ./scenarios

RUN chown -R cono_sandbox:cono_sandbox /app
USER cono_sandbox

ENV PYTHONPATH=/app/packages
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "unittest", "discover", "-s", "scenarios", "-v"]
