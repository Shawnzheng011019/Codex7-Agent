"""
System prompt generation mechanism.
Creates dynamic prompts based on context, tools, and environment.
"""

import os
import platform
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging


class PromptGenerator:
    """Dynamic system prompt generator."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_system_prompt(
        self,
        workspace_path: str,
        session_context: Dict[str, Any],
        available_tools: List[str],
        user_preferences: Dict[str, Any] = None
    ) -> str:
        """
        Generate comprehensive system prompt based on environment and context.
        
        Args:
            workspace_path: Current workspace directory
            session_context: Current session information
            available_tools: List of available tools
            user_preferences: User-specific preferences
            
        Returns:
            Complete system prompt
        """
        
        # Collect environment information
        env_info = self._collect_environment_info(workspace_path)
        
        # Generate identity section
        identity = self._generate_identity_section()
        
        # Generate security policy
        security_policy = self._generate_security_policy()
        
        # Generate tool guidance
        tool_guidance = self._generate_tool_guidance(available_tools)
        
        # Generate style guidelines
        style_guidelines = self._generate_style_guidelines()
        
        # Generate context-specific guidance
        context_guidance = self._generate_context_guidance(session_context)
        
        # Combine all sections
        prompt_parts = [
            identity,
            security_policy,
            tool_guidance,
            style_guidelines,
            context_guidance,
            self._generate_final_instructions()
        ]
        
        return "\n\n".join(prompt_parts)
    
    def _collect_environment_info(self, workspace_path: str) -> Dict[str, Any]:
        """Collect comprehensive environment information."""
        info = {
            "workspace_path": workspace_path,
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "current_time": self._get_current_time(),
            "git_info": self._get_git_info(workspace_path),
            "directory_structure": self._get_directory_structure(workspace_path)
        }
        
        return info
    
    def _get_current_time(self) -> str:
        """Get current time in readable format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_git_info(self, workspace_path: str) -> Dict[str, Any]:
        """Get Git repository information."""
        try:
            os.chdir(workspace_path)
            
            # Check if it's a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Get branch name
                branch_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True
                )
                branch = branch_result.stdout.strip()
                
                # Get last commit
                commit_result = subprocess.run(
                    ["git", "log", "-1", "--oneline"],
                    capture_output=True,
                    text=True
                )
                last_commit = commit_result.stdout.strip()
                
                # Get uncommitted changes
                status_result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True
                )
                changes = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0
                
                return {
                    "is_git_repo": True,
                    "branch": branch,
                    "last_commit": last_commit,
                    "uncommitted_changes": changes
                }
            else:
                return {"is_git_repo": False}
                
        except Exception as e:
            self.logger.warning(f"Failed to get git info: {e}")
            return {"is_git_repo": False, "error": str(e)}
    
    def _get_directory_structure(self, workspace_path: str) -> Dict[str, Any]:
        """Get basic directory structure information."""
        try:
            workspace = Path(workspace_path)
            
            # Count files by extension
            file_counts = {}
            total_files = 0
            
            for item in workspace.rglob("*"):
                if item.is_file():
                    total_files += 1
                    ext = item.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
            
            # Get top-level directories
            directories = [
                str(d.name) for d in workspace.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ]
            
            return {
                "total_files": total_files,
                "file_types": dict(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                "top_directories": directories[:5]
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get directory structure: {e}")
            return {"error": str(e)}
    
    def _generate_identity_section(self) -> str:
        """Generate agent identity section."""
        return """You are Claude Code, Anthropic's official CLI for Claude.

You are an interactive CLI tool that helps users with software engineering tasks. Your role is to assist with defensive security tasks only - you refuse to create, modify, or improve code that may be used maliciously. You allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation."""
    
    def _generate_security_policy(self) -> str:
        """Generate security policy section."""
        return """SECURITY POLICY:
- NEVER create, modify, or improve code that may be used maliciously
- Allow security analysis, detection rules, vulnerability explanations, defensive tools, and security documentation
- NEVER generate or guess URLs unless confident they help with programming
- When accessing web content, use URLs provided by the user or local files only
- Always prioritize defensive security over offensive capabilities"""
    
    def _generate_tool_guidance(self, available_tools: List[str]) -> str:
        """Generate tool usage guidance."""
        tool_categories = {
            "filesystem": ["read_file", "write_file", "list_directory", "glob"],
            "search": ["grep", "find", "search"],
            "code": ["analyze_code", "format_code", "lint"],
            "system": ["bash", "shell"],
            "util": ["task", "todo", "note"]
        }
        
        tool_guidance = ["TOOL USAGE GUIDELINES:"]
        
        for category, tools in tool_categories.items():
            available_in_category = [t for t in tools if t in available_tools]
            if available_in_category:
                tool_guidance.append(f"- {category.upper()}: {', '.join(available_in_category)}")
        
        tool_guidance.extend([
            "- Use appropriate tools based on task complexity",
            "- Prefer concurrency-safe tools when possible",
            "- Always validate tool parameters before execution",
            "- Provide clear explanations for tool usage decisions"
        ])
        
        return "\n".join(tool_guidance)
    
    def _generate_style_guidelines(self) -> str:
        """Generate style and communication guidelines."""
        return """STYLE GUIDELINES:
- Be concise, direct, and to the point
- Use GitHub-flavored markdown for formatting
- Explain non-trivial commands and their purpose
- Minimize output tokens while maintaining clarity
- Follow existing code conventions and patterns
- NEVER add comments unless explicitly requested
- Provide helpful alternatives when refusing requests
- Use file_path:line_number format for code references"""
    
    def _generate_context_guidance(self, session_context: Dict[str, Any]) -> str:
        """Generate context-specific guidance."""
        guidance_parts = ["CONTEXT-SPECIFIC GUIDANCE:"]
        
        # Task management
        if session_context.get("use_todo", True):
            guidance_parts.append(
                "- Use TodoWrite tool proactively for complex multi-step tasks"
                "- Update task status in real-time as you work"
                "- Complete current tasks before starting new ones"
            )
        
        # Testing guidance
        if session_context.get("run_tests", True):
            guidance_parts.append(
                "- Run tests and build process after implementing changes"
                "- Address any failures or errors immediately"
                "- Ensure tests and build pass before considering task complete"
            )
        
        # Git workflow
        if session_context.get("git_repo", False):
            guidance_parts.append(
                "- Check git status before making changes"
                "- Commit changes with descriptive messages"
                "- NEVER commit sensitive information or secrets"
            )
        
        return "\n".join(guidance_parts)
    
    def _generate_final_instructions(self) -> str:
        """Generate final operational instructions."""
        return """FINAL INSTRUCTIONS:
- Always understand the codebase before making changes
- Use search tools extensively to understand existing patterns
- Follow security best practices in all code
- Provide actionable, tested solutions
- When unsure, ask for clarification rather than guessing
- Maintain consistency with existing code style and architecture"""
    
    def generate_task_specific_prompt(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate prompt for specific task types."""
        prompts = {
            "search": """SEARCH TASK GUIDANCE:
- Use multiple search approaches (grep, find, file listing)
- Search for patterns, not just exact matches
- Consider case-insensitive searches by default
- Look in relevant file types based on the search context""",
            
            "analyze": """ANALYSIS TASK GUIDANCE:
- Read relevant files thoroughly before analysis
- Consider both code structure and business logic
- Identify potential issues, improvements, and patterns
- Provide specific, actionable recommendations""",
            
            "create": """CREATION TASK GUIDANCE:
- Follow existing project structure and conventions
- Create comprehensive, well-tested implementations
- Include appropriate documentation and error handling
- Ensure new code integrates well with existing codebase""",
            
            "modify": """MODIFICATION TASK GUIDANCE:
- Understand current behavior before making changes
- Make minimal, targeted changes
- Test changes thoroughly
- Ensure backward compatibility where possible""",
            
            "debug": """DEBUGGING GUIDANCE:
- Reproduce the issue before attempting fixes
- Use systematic debugging approaches
- Add logging or debugging output as needed
- Verify fixes resolve the issue completely"""
        }
        
        return prompts.get(task_type, "Apply general best practices for the task")
    
    def generate_tool_specific_prompt(self, tool_name: str) -> str:
        """Generate prompt for specific tool usage."""
        tool_prompts = {
            "read_file": "When reading files, focus on relevant sections and provide context about what you're looking for.",
            "write_file": "When writing files, ensure content is complete, well-formatted, and includes necessary imports/dependencies.",
            "bash": "When executing bash commands, explain what each command does and verify results.",
            "task": "Use the Task tool for complex, multi-faceted problems that require parallel analysis."
        }
        
        return tool_prompts.get(tool_name, "Use the tool appropriately for the task at hand")
    
    def refresh_prompt(self, **kwargs) -> str:
        """Regenerate prompt with updated context."""
        return self.generate_system_prompt(**kwargs)