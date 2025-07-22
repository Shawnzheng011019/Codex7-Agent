"""
CLI entry point for Codex7 agent to work with SWE-bench evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .agent.orchestrator import AgentConfig, create_agent


def main():
    """Main entry point for Codex7 agent CLI."""
    parser = argparse.ArgumentParser(description="Codex7 Agent for SWE-bench evaluation")
    parser.add_argument("--file", required=True, help="Path to problem statement file")
    parser.add_argument("--working-dir", required=True, help="Working directory for the task")
    parser.add_argument("--config-file", required=True, help="Path to configuration file")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum number of steps")
    parser.add_argument("--must-patch", action="store_true", help="Must generate a patch")
    parser.add_argument("--patch-path", help="Path to save generated patch")
    parser.add_argument("--trajectory-file", help="Path to save trajectory JSON")
    
    args = parser.parse_args()
    
    # Read problem statement
    with open(args.file, 'r') as f:
        problem_statement = f.read()
    
    # Read configuration
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)
    
    # Setup working directory
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(working_dir)
    
    # Create agent configuration
    config = AgentConfig(
        max_steps=args.max_steps,
        working_dir=str(working_dir),
        **config_data
    )
    
    # Create and run agent
    agent = create_agent(config)
    
    print(f"Starting Codex7 agent with problem: {problem_statement[:100]}...")
    
    try:
        result = agent.run(problem_statement)
        
        # Save trajectory if requested
        if args.trajectory_file:
            trajectory_data = {
                "problem_statement": problem_statement,
                "steps": result.steps if hasattr(result, 'steps') else [],
                "final_patch": result.patch if hasattr(result, 'patch') else "",
                "success": result.success if hasattr(result, 'success') else False
            }
            
            with open(args.trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
        
        # Save patch if requested
        if args.patch_path and hasattr(result, 'patch') and result.patch:
            with open(args.patch_path, 'w') as f:
                f.write(result.patch)
            print(f"Patch saved to {args.patch_path}")
        elif args.must_patch:
            print("Warning: No patch generated but --must-patch was specified")
            sys.exit(1)
        
        print("Codex7 agent completed successfully")
        
    except Exception as e:
        print(f"Error running Codex7 agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()