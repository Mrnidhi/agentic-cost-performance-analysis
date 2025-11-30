#!/usr/bin/env python3
"""
AI Agent Performance Intelligence System - Local Development Runner
Course: DATA 230 (Data Visualization) at SJSU

Usage:
    python run_api.py                    # Run with defaults
    python run_api.py --port 8080        # Custom port
    python run_api.py --reload           # Enable hot reload
    python run_api.py --debug            # Debug mode
"""

import argparse
import os
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Run AI Agent Performance Intelligence API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)"
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        args.log_level = "debug"
        args.reload = True

    print("=" * 60)
    print("AI Agent Performance Intelligence API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 60)
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()

