"""Command-line tasks.

    python manage.py check-config          # validate .env without printing any secret
    python manage.py init-db [--wait 60]   # create/upgrade checkpoint + application tables (repeatable)
    python manage.py eval [options]        # evaluation experiments (see `eval --help`)
"""

import argparse
import sys

import config
import memory


def check_config(_args) -> int:
    ok = True
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        state = "missing or placeholder" if config.get(name) is None else "set"
        ok &= state == "set"
        print(f"  {name:<22} {state}")
    try:
        memory.database_url()
        print(f"  {'DATABASE_URL':<22} set")
    except config.ConfigError as exc:
        ok = False
        print(f"  {'DATABASE_URL':<22} invalid: configure {', '.join(exc.names)}")
    lf = config.langfuse_settings()
    print(f"  {'Langfuse tracing':<22} {'enabled → ' + lf.base_url if lf.enabled else lf.reason}")
    print("  Guardrails AI          local validators (no key needed)")
    if not ok:
        print("\nFill the missing values in .env (see .env.example for where each comes from).")
    return 0 if ok else 1


def init_db(args) -> int:
    try:
        url = memory.database_url()
        memory.wait_for_database(url, timeout=args.wait)
        pool = memory.open_pool(url, max_size=2)
        for line in memory.init_db(pool):
            print(f"  ✓ {line}")
        pool.close()
    except (config.ConfigError, memory.DatabaseError) as exc:
        print(f"init-db failed:\n{exc}", file=sys.stderr)
        return 1
    print("Database ready.")
    return 0


def run_eval(args) -> int:
    from experiments import main as eval_main

    return eval_main(args)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not be UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="manage.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config", help="validate configuration").set_defaults(fn=check_config)
    p = sub.add_parser("init-db", help="create or upgrade database tables")
    p.add_argument("--wait", type=float, default=60, help="seconds to wait for the database (default 60)")
    p.set_defaults(fn=init_db)
    from experiments import add_arguments

    add_arguments(sub.add_parser("eval", help="run an evaluation experiment")).set_defaults(fn=run_eval)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
