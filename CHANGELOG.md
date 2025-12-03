# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial open source release
- Field projection capabilities (include/exclude/view modes)
- Grep search functionality for filtering tool outputs
- **Context lines support for grep** - Include lines before/after matches (similar to grep -A, -B, -C)
- **Multiline pattern support for grep** - Enable patterns that span multiple lines
- Support for multiple underlying MCP servers
- Token savings tracking and reporting
- Configuration via YAML file
- Comprehensive documentation

### Changed
- Restructured repository for open source standards
- Moved source code to `src/mcp_proxy/` package structure
- Organized tests into `tests/` directory
- Enhanced grep processor with context lines and multiline matching

## [0.1.0] - 2024-XX-XX

### Added
- Initial release
- MCP Proxy Server implementation
- Field projection processor
- Grep processor
- Server connection management
- Tool aggregation from multiple servers
- Configuration loading from YAML

[Unreleased]: https://github.com/yourusername/mcp-proxy-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/mcp-proxy-server/releases/tag/v0.1.0

