"""
Configuration loading utilities with Pydantic validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mcp_proxy.logging_config import get_logger

logger = get_logger(__name__)


class ServerConfig(BaseModel):
    """
    Configuration model for a single underlying MCP server.

    Attributes:
        name: Unique identifier for the server (used in tool names as "name_tool_name")
        command: Command to run the server (e.g., "npx", "python", "node", "uv")
        args: List of arguments to pass to the command
        env: Optional environment variables dictionary
    """

    name: str = Field(
        ...,
        description="Unique identifier for the server",
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    command: str = Field(
        ...,
        description="Command to run the server",
        min_length=1,
    )
    args: List[str] = Field(
        default_factory=list,
        description="List of arguments to pass to the command",
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional environment variables for the server process",
    )

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and trim server name."""
        if not isinstance(v, str):
            return v
        # Trim whitespace first
        v = v.strip()
        if not v:
            raise ValueError("Server name cannot be empty or whitespace only")
        return v

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command is not empty."""
        if not v.strip():
            raise ValueError("Command cannot be empty or whitespace only")
        return v.strip()

    @field_validator("args")
    @classmethod
    def validate_args(cls, v: List[str]) -> List[str]:
        """Validate args list."""
        # Allow empty args list (some servers don't need arguments)
        return v if v is not None else []

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "filesystem",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
            }
        }
    )


class ProxyConfig(BaseModel):
    """
    Root configuration model for the MCP Proxy Server.

    Attributes:
        underlying_servers: List of server configurations
    """

    underlying_servers: List[ServerConfig] = Field(
        default_factory=list,
        description="List of underlying MCP server configurations",
    )

    @field_validator("underlying_servers")
    @classmethod
    def validate_unique_names(cls, v: List[ServerConfig]) -> List[ServerConfig]:
        """Ensure all server names are unique."""
        names = [server.name for server in v]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(
                f"Duplicate server names found: {', '.join(set(duplicates))}. "
                "Each server must have a unique name."
            )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "underlying_servers": [
                    {
                        "name": "filesystem",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    }
                ]
            }
        }
    )


def load_config(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """
    Load and validate server configurations from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        List of validated server configurations as dictionaries

    Raises:
        ValidationError: If configuration validation fails
        FileNotFoundError: If config file doesn't exist (logged as warning, returns empty list)
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found. Using empty configuration.")
            return []

        with open(config_file, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # Handle empty or None config
        if raw_config is None:
            logger.info("Config file is empty. Using empty configuration.")
            return []

        # Validate configuration with Pydantic
        try:
            proxy_config = ProxyConfig.model_validate(raw_config)
            logger.debug(f"Successfully loaded and validated {len(proxy_config.underlying_servers)} server(s)")

            # Convert Pydantic models to dictionaries for backward compatibility
            return [server.model_dump() for server in proxy_config.underlying_servers]

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            # Provide helpful error messages
            errors = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_msg = error["msg"]
                errors.append(f"  {field_path}: {error_msg}")

            error_message = "Configuration validation errors:\n" + "\n".join(errors)
            logger.error(error_message)
            # Re-raise with better context as ValueError
            raise ValueError(f"Invalid configuration in {config_path}:\n{error_message}") from e

    except FileNotFoundError:
        # Already handled above, but keep for safety
        logger.warning(f"Config file {config_path} not found. Using empty configuration.")
        return []
    except (ValidationError, ValueError):
        # Re-raise validation errors (ValueError is our wrapped ValidationError)
        raise
    except yaml.YAMLError as e:
        error_msg = f"YAML parsing error in {config_path}: {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error loading config from {config_path}: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

