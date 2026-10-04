# NavDP / learned-baseline integration — decision & progress log

Maintained by the agent while working autonomously on branch `feat/navdp-baselines`.
**Review the "Needs your review" items first.** Nothing here has been pushed. Delete this file once
you've read it (it is committed on the feature branch only).

Plan of record: `~/.claude/plans/1-it-s-fine-to-goofy-gizmo.md` (supersedes
`baseline_integration_plan.md`).

## Needs your review

_(none yet)_

## Decisions made (with rationale)

| # | Decision | Why | Reversible by |
|---|---|---|---|
| D1 | NavDP fork pinned at `8b9ee13` (`Herschenglime/NavDP`, branch `nav-arena`); labmate's untracked harness NOT put on the fork | Fork holds only upstream-file fixes; harness is ported into nav_arena | — |
| D2 | Dependencies installed into `env_isaaclab` via `uv pip --no-deps`, incl. labmate's prebuilt mmcv wheel copied from `alex-spark`; pinned in `requirements/baselines.txt` | Keeps Isaac's torch/numpy untouched; wheel imports and `mmcv.ops` loads here | `uv pip uninstall` |
| D3 | Added `RobotEmbodimentCfg.body_link` (USD link carrying colliders/sensors): `chassis_link` Carter, `base_link` Dingo. `chassis_frame` stays a TF/URDF frame name | Dingo USD has a single rigid body; hardcoded `/Robot/chassis_link` paths would be wrong for it. (Plan called this `contact_body`.) | — |
| D4 | `dingo_stage_patch` rewritten in plain USD (no `pxr.PhysxSchema`) and raises if no caster material found | PhysxSchema only exists inside Kit; old code silently skipped the friction fix | — |
| D5 | Camera uses `IsaacRtxRendererCfg` explicitly | Labmate validated this; Lab's default is the generic `RendererCfg` | one line |
| D6 | Timing: keep nav_arena's dt=0.01 / decimation=2 (50 Hz control); only make `render_interval` configurable (20 → 5 Hz camera for baselines) | User: perfect replication not required | — |
| D7 | Only InteriorAgent scenes (no Nucleus office) | User decision | — |

## Progress

- [x] Phase 1 (embodiment registry + Dingo) — `6fa7f73`, follow-up `e929347`
- [x] Paths refactor — `4d42904`
- [x] Dependencies + pinned requirements — `e8dc139`
- [x] Phase 2 — RGB-D camera + multi-robot verify_embodiment (headless run: Dingo+camera PASS, Carter PASS)
- [ ] Phase 3 — methods restructure + NavDP adapter port
- [ ] Phase 4 — generalized PointNavTask
- [ ] Phase 5 — episode runner, verify_baseline, Dingo integration test

## Findings / surprises

- **Phase 2 / boot-order trap:** `verify_embodiment.py --robot` cannot use argparse `choices=list_embodiments()`:
  importing `nav_arena.embodiments` at module level pulls in `pxr` before `AppLauncher` boots and the app crashes
  (`SystemError: Missing frame when calling profile function`). The name is validated after boot instead.
  Any future script that needs the registry for CLI choices has the same constraint.
- **Phase 2 / camera sanity:** Dingo nearest ground return 1.12 m, depth valid fraction 0.43, RGB uint8
  `(1,360,640,3)`, depth `(1,360,640,1)` float32. Matches the expected geometry (camera ~0.42 m above floor,
  20.6 deg half-VFOV). The Dingo drove 0.139 m in 60 steps at v=0.5 (Carter 0.159 m) — acceleration-limited
  at start, not a caster-drag signal; no ground-truth friction test exists yet (see Phase 5 integration test).
