"""
Processors for field projection and grep operations.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from mcp.types import Content, ImageContent, TextContent


class ProjectionProcessor:
    """Handles field projection operations on tool responses."""

    @staticmethod
    def apply_projection(
        data: Any, projection: Dict[str, Any], path: str = ""
    ) -> Any:
        """
        Apply field projection to data based on projection spec.

        Args:
            data: The data to project (dict, list, or primitive)
            projection: Projection specification with 'mode' and 'fields'
            path: Current JSON path for nested structures

        Returns:
            Projected data
        """
        mode = projection.get("mode", "include")
        fields = projection.get("fields", [])

        # Handle lists by applying projection to each item
        if isinstance(data, list):
            return [
                ProjectionProcessor.apply_projection(item, projection, path)
                for item in data
            ]

        # Handle dictionaries
        if not isinstance(data, dict):
            return data

        if mode == "include":
            # Group fields by parent path for arrays (e.g., "users.name", "users.email" -> "users": ["name", "email"])
            array_projections = {}  # parent_key -> list of nested fields
            regular_fields = []
            
            for field in fields:
                if "." in field:
                    parts = field.split(".")
                    parent_key = parts[0]
                    nested_field = ".".join(parts[1:])
                    
                    # Check if parent is an array
                    if parent_key in data and isinstance(data[parent_key], list):
                        if parent_key not in array_projections:
                            array_projections[parent_key] = []
                        array_projections[parent_key].append(nested_field)
                    else:
                        regular_fields.append(field)
                else:
                    regular_fields.append(field)
            
            result = {}
            
            # Handle array projections (e.g., "users": ["name", "email"])
            for parent_key, nested_fields in array_projections.items():
                if parent_key in data:
                    nested_projection = {
                        "mode": "include",
                        "fields": nested_fields
                    }
                    result[parent_key] = ProjectionProcessor.apply_projection(
                        data[parent_key], nested_projection, parent_key
                    )
            
            # Handle regular fields (direct keys or nested dict paths)
            for field in regular_fields:
                if field in data:
                    value = data[field]
                    # Recursively apply projection to nested objects
                    if isinstance(value, (dict, list)):
                        result[field] = ProjectionProcessor.apply_projection(
                            value, projection, f"{path}.{field}" if path else field
                        )
                    else:
                        result[field] = value
                # Handle nested paths like "user.name" (for dicts, not arrays)
                elif "." in field:
                    parts = field.split(".")
                    current = data
                    # Navigate to the nested field
                    for part in parts[:-1]:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            break
                    else:
                        if isinstance(current, dict) and parts[-1] in current:
                            # Create nested structure preserving hierarchy
                            nested = result
                            for part in parts[:-1]:
                                if part not in nested:
                                    nested[part] = {}
                                nested = nested[part]
                            value = current[parts[-1]]
                            # Recursively apply projection to nested values
                            if isinstance(value, (dict, list)):
                                nested[parts[-1]] = ProjectionProcessor.apply_projection(
                                    value, projection, field
                                )
                            else:
                                nested[parts[-1]] = value
            return result

        elif mode == "exclude":
            # Exclude specified fields
            result = {}
            for key, value in data.items():
                # Check if this key should be excluded directly
                if key in fields:
                    continue
                
                # Check if this key is part of a nested exclusion path
                # e.g., if excluding "user.password", we keep "user" but exclude "password" inside it
                should_exclude_key = False
                nested_exclusions = []
                for field in fields:
                    if field == key:
                        should_exclude_key = True
                        break
                    elif field.startswith(f"{key}."):
                        # This is a nested exclusion - keep the key but exclude nested field
                        nested_exclusions.append(field[len(key) + 1:])  # Remove "key." prefix
                
                if should_exclude_key:
                    continue
                
                # If there are nested exclusions, apply them recursively
                if nested_exclusions:
                    nested_projection = {
                        "mode": "exclude",
                        "fields": nested_exclusions
                    }
                    if isinstance(value, (dict, list)):
                        result[key] = ProjectionProcessor.apply_projection(
                            value, nested_projection, f"{path}.{key}" if path else key
                        )
                    else:
                        result[key] = value
                else:
                    # No exclusions for this key - include it
                    # Recursively apply exclusion to nested structures
                    if isinstance(value, (dict, list)):
                        result[key] = ProjectionProcessor.apply_projection(
                            value, projection, f"{path}.{key}" if path else key
                        )
                    else:
                        result[key] = value
            
            return result

        elif mode == "view":
            # Named preset view (simplified - could be extended)
            view_name = projection.get("view", fields[0] if fields else "default")
            # For now, treat as include
            return ProjectionProcessor.apply_projection(
                data, {"mode": "include", "fields": fields}
            )

        return data

    @staticmethod
    def project_content(
        content: List[Content], projection: Dict[str, Any]
    ) -> List[Content]:
        """
        Apply projection to MCP Content objects.

        Args:
            content: List of Content objects
            projection: Projection specification

        Returns:
            List of projected Content objects
        """
        projected = []
        for item in content:
            if isinstance(item, TextContent):
                # Try to parse as JSON for structured content
                try:
                    data = json.loads(item.text)
                    if isinstance(data, dict):
                        projected_data = ProjectionProcessor.apply_projection(
                            data, projection
                        )
                        projected.append(
                            TextContent(
                                type="text",
                                text=json.dumps(projected_data, indent=2),
                            )
                        )
                    else:
                        projected.append(item)
                except json.JSONDecodeError:
                    # Not JSON, return as-is (can't project plain text)
                    projected.append(item)
            elif isinstance(item, ImageContent):
                # Images can't be projected
                projected.append(item)
            else:
                projected.append(item)
        return projected


class GrepProcessor:
    """Handles grep-like search operations on tool outputs."""

    @staticmethod
    def apply_grep(
        content: List[Content], grep_spec: Dict[str, Any]
    ) -> List[Content]:
        """
        Apply grep search to content.

        Args:
            content: List of Content objects
            grep_spec: Grep specification with 'pattern', 'caseInsensitive', etc.

        Returns:
            Filtered Content objects
        """
        pattern = grep_spec.get("pattern", "")
        case_insensitive = grep_spec.get("caseInsensitive", False)
        max_matches = grep_spec.get("maxMatches")
        target = grep_spec.get("target", "content")  # 'content' or 'structuredContent'
        multiline = grep_spec.get("multiline", False)
        
        # Context lines configuration
        context_lines = grep_spec.get("contextLines", {})
        context_before = context_lines.get("before", 0) if isinstance(context_lines, dict) else 0
        context_after = context_lines.get("after", 0) if isinstance(context_lines, dict) else 0
        context_both = context_lines.get("both", 0) if isinstance(context_lines, dict) else 0
        
        # If 'both' is specified, it overrides 'before' and 'after'
        if context_both > 0:
            context_before = context_both
            context_after = context_both

        if not pattern:
            return content

        # Validate and compile regex pattern
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            if multiline:
                flags |= re.MULTILINE | re.DOTALL
            regex = re.compile(pattern, flags)
        except re.error as e:
            # Invalid regex pattern - return error message as content
            return [
                TextContent(
                    type="text",
                    text=f"Error: Invalid regex pattern '{pattern}': {str(e)}",
                )
            ]

        filtered = []
        match_count = 0

        for item in content:
            if isinstance(item, TextContent):
                text = item.text

                # Try to parse as JSON for structured content search
                if target == "structuredContent":
                    try:
                        data = json.loads(text)
                        # Search in structured data
                        matches = GrepProcessor._search_in_structure(
                            data, regex, max_matches, match_count
                        )
                        if matches is not None and matches != {} and matches != []:
                            filtered.append(
                                TextContent(
                                    type="text",
                                    text=json.dumps(matches, indent=2),
                                )
                            )
                            # Count matches more accurately
                            if isinstance(matches, list):
                                match_count += len(matches)
                            elif isinstance(matches, dict):
                                match_count += GrepProcessor._count_dict_matches(matches)
                            else:
                                match_count += 1
                    except json.JSONDecodeError:
                        # Not JSON, fall back to text search
                        matches, match_lines = GrepProcessor._search_in_text(
                            text, regex, max_matches, match_count, context_before, context_after, multiline
                        )
                        if matches:
                            filtered.append(TextContent(type="text", text=matches))
                            match_count += match_lines
                else:
                    # Plain text search
                    matches, match_lines = GrepProcessor._search_in_text(
                        text, regex, max_matches, match_count, context_before, context_after, multiline
                    )
                    if matches:
                        filtered.append(TextContent(type="text", text=matches))
                        match_count += match_lines

            elif isinstance(item, ImageContent):
                # Can't grep images - optionally skip or include
                # For now, we skip images when grep is applied
                pass

            if max_matches and match_count >= max_matches:
                break

        return filtered if filtered else [TextContent(type="text", text="No matches found.")]

    @staticmethod
    def _search_in_text(
        text: str, 
        regex: re.Pattern, 
        max_matches: Optional[int], 
        current_count: int,
        context_before: int = 0,
        context_after: int = 0,
        multiline: bool = False
    ) -> Tuple[str, int]:
        """
        Search for pattern in text, returning matching lines with optional context.
        
        Args:
            text: Text to search in
            regex: Compiled regex pattern
            max_matches: Maximum number of matches to return
            current_count: Current match count
            context_before: Number of lines to include before each match
            context_after: Number of lines to include after each match
            multiline: Whether pattern can span multiple lines
            
        Returns:
            Tuple of (result_string, number_of_actual_matches)
        """
        # For multiline patterns, search the entire text as one block
        if multiline:
            matches = list(regex.finditer(text))
            if not matches:
                return "", 0
            
            # Limit matches if needed
            if max_matches:
                remaining_matches = max_matches - current_count
                if remaining_matches <= 0:
                    return "", 0
                matches = matches[:remaining_matches]
            
            # For multiline, return the matched text portions
            result_parts = []
            for match in matches:
                match_text = match.group(0)
                result_parts.append(match_text)
            
            return "\n---\n".join(result_parts), len(matches)
        
        # For single-line patterns, use line-by-line search
        lines = text.split("\n")
        matched_line_indices = set()
        
        # First pass: find all matching line indices
        for i, line in enumerate(lines):
            if regex.search(line):
                matched_line_indices.add(i)
        
        if not matched_line_indices:
            return "", 0
        
        # Limit matches if needed
        if max_matches:
            remaining_matches = max_matches - current_count
            if remaining_matches <= 0:
                return "", 0
            # Take only the first N matches
            matched_line_indices = set(sorted(matched_line_indices)[:remaining_matches])
        
        # Second pass: collect matches with context, avoiding duplicates
        included_indices = set()
        result_lines = []
        actual_match_count = 0
        
        for match_idx in sorted(matched_line_indices):
            # Add context before
            for i in range(max(0, match_idx - context_before), match_idx):
                if i not in included_indices:
                    result_lines.append(lines[i])
                    included_indices.add(i)
            
            # Add the matching line
            if match_idx not in included_indices:
                result_lines.append(lines[match_idx])
                included_indices.add(match_idx)
                actual_match_count += 1
            
            # Add context after
            for i in range(match_idx + 1, min(len(lines), match_idx + 1 + context_after)):
                if i not in included_indices:
                    result_lines.append(lines[i])
                    included_indices.add(i)
            
            # Add separator between match groups if context is used and matches are far apart
            if context_before > 0 or context_after > 0:
                # Check if next match is far enough to add separator
                next_match = min([m for m in matched_line_indices if m > match_idx], default=None)
                if next_match and next_match > match_idx + context_after + 1:
                    # Add separator line
                    result_lines.append("---")
        
        return "\n".join(result_lines), actual_match_count

    @staticmethod
    def _search_in_structure(
        data: Any, regex: re.Pattern, max_matches: Optional[int], current_count: int
    ) -> Any:
        """Recursively search for pattern in structured data."""
        if max_matches and current_count >= max_matches:
            return None

        if isinstance(data, dict):
            matches = {}
            count = current_count
            for key, value in data.items():
                if max_matches and count >= max_matches:
                    break
                # Search in key
                key_matches = regex.search(str(key))
                # Search in value (if string)
                value_matches = isinstance(value, str) and regex.search(value)
                
                if key_matches or value_matches:
                    # Include the entire key-value pair if either matches
                    matches[key] = value
                    count += 1
                elif isinstance(value, (dict, list)):
                    # Recursively search nested structures
                    nested = GrepProcessor._search_in_structure(
                        value, regex, max_matches, count
                    )
                    if nested is not None and nested != {} and nested != []:
                        matches[key] = nested
                        count += 1
            return matches if matches else None
            
        elif isinstance(data, list):
            matches = []
            count = current_count
            for item in data:
                if max_matches and count >= max_matches:
                    break
                if isinstance(item, (dict, list)):
                    nested = GrepProcessor._search_in_structure(
                        item, regex, max_matches, count
                    )
                    if nested is not None and nested != {} and nested != []:
                        matches.append(nested)
                        count += 1
                elif isinstance(item, str) and regex.search(item):
                    matches.append(item)
                    count += 1
            return matches if matches else None
        else:
            # Primitive value - check if it matches
            if regex.search(str(data)):
                return data
            return None

    @staticmethod
    def _count_dict_matches(matches: Dict[str, Any]) -> int:
        """Count the number of matches in a dictionary structure."""
        count = 0
        for value in matches.values():
            if isinstance(value, dict):
                count += GrepProcessor._count_dict_matches(value)
            elif isinstance(value, list):
                count += len(value)
            else:
                count += 1
        return count

