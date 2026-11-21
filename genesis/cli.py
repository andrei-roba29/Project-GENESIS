"""Headless CLI for Project GENESIS.

Examples:
  python -m genesis.cli generate --seed 42 --preset earthlike
  python -m genesis.cli generate --seed 7 --width 360 --height 180 --tick 365
  python -m genesis.cli serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import os

from .config import PlanetConfig
from .engine import Engine
from . import persistence


def cmd_generate(args) -> int:
    cfg = PlanetConfig(seed=args.seed, preset=args.preset, width=args.width,
                       height=args.height, world_name=args.name)
    engine = Engine(cfg)
    report = engine.generate()
    print(json.dumps(report, indent=2))
    if args.tick:
        print(f"\n--- ticking {args.tick} days ---")
        print(json.dumps(engine.tick_days(args.tick), indent=2))
        print(json.dumps(engine.planet.summary(), indent=2))
    if args.save:
        path = persistence.save_world(engine.planet, engine.fauna, engine.clock, "saves", args.save)
        print(f"Saved -> {path}")
    return 0


def cmd_serve(args) -> int:
    from .server import main
    main(port=args.port)
    return 0


def cmd_load(args) -> int:
    path = args.file if os.path.isfile(args.file) else os.path.join("saves", args.file + ".json")
    planet, fauna_obj, clock = persistence.load_world(path)
    print(json.dumps(planet.summary(), indent=2))
    print(f"Species in radius: {len(fauna_obj.species_in_radius())}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="genesis", description="Project GENESIS — Phase 1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="Generate a world (headless)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--preset", default="earthlike",
                   choices=["earthlike", "archipelago", "desert", "frozen", "custom"])
    g.add_argument("--width", type=int, default=180)
    g.add_argument("--height", type=int, default=90)
    g.add_argument("--name", default="Genesis World")
    g.add_argument("--tick", type=float, default=0, help="Simulated days to advance after generation")
    g.add_argument("--save", default="", help="Save name (saves/<name>.json)")
    g.set_defaults(fn=cmd_generate)
    s = sub.add_parser("serve", help="Run the local web server")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    l = sub.add_parser("load", help="Inspect a saved world")
    l.add_argument("file", help="Save name or path")
    l.set_defaults(fn=cmd_load)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
