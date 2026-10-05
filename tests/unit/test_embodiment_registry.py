# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RobotEmbodiment contract, registry, and embodiment configs."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

import nav_arena.embodiments as embodiments
from nav_arena.embodiments import (
    DINGO_ACTION_CFG,
    NOVA_CARTER_ACTION_CFG,
    NOVA_CARTER_CFG,
    DingoEmbodimentCfg,
    NovaCarterEmbodimentCfg,
    RobotEmbodimentCfg,
    clear_registry,
    generate_minimal_urdf,
    get_embodiment,
    list_embodiments,
    register_default_embodiments,
    register_embodiment,
)


@pytest.fixture(autouse=True)
def restore_registry_after_test():
    """Ensure registry defaults are restored after every test."""
    yield
    clear_registry()
    register_default_embodiments()


def test_default_embodiments_registered():
    """Verify that default embodiments ('nova_carter', 'dingo', 'kaya') are registered."""
    names = list_embodiments()
    assert "nova_carter" in names
    assert "dingo" in names
    assert "kaya" in names


def test_get_nova_carter_embodiment():
    """Verify Nova Carter geometry, frame, and kinematic retrieval."""
    cfg = get_embodiment("nova_carter")
    assert isinstance(cfg, RobotEmbodimentCfg)
    assert isinstance(cfg, NovaCarterEmbodimentCfg)
    assert cfg.name == "nova_carter"
    assert cfg.wheel_radius == 0.14
    assert cfg.wheel_base == 0.413
    assert cfg.max_linear_speed == 2.0
    assert cfg.max_angular_speed == 3.0
    assert cfg.base_frame == "base_link"
    assert cfg.chassis_frame == "chassis_link"
    assert cfg.body_link == "chassis_link"
    assert cfg.lidar_frame == "lidar_link"
    assert cfg.camera_frame == "camera_link"
    assert cfg.sensor_height == 0.35
    assert cfg.lidar_offset == (0.0, 0.0, 0.35)
    assert cfg.robot_radius == 0.28
    assert cfg.robot_height == 0.40
    assert cfg.articulation_cfg is not None
    assert cfg.action_cfg is not None
    assert cfg.action_cfg.wheel_radius == 0.14
    assert cfg.action_cfg.wheel_base == 0.413


def test_get_dingo_embodiment():
    """Verify Clearpath Dingo geometry, frame, kinematics, and stage patch retrieval."""
    cfg = get_embodiment("dingo")
    assert isinstance(cfg, RobotEmbodimentCfg)
    assert isinstance(cfg, DingoEmbodimentCfg)
    assert cfg.name == "dingo"
    assert cfg.wheel_radius == 0.1225
    assert cfg.wheel_base == 0.4523232
    assert cfg.max_linear_speed == 2.0
    assert cfg.max_angular_speed == 3.0
    assert cfg.base_frame == "base_link"
    assert cfg.chassis_frame == "chassis_link"
    # The Dingo USD has one rigid body; colliders and sensors attach to base_link.
    assert cfg.body_link == "base_link"
    assert cfg.lidar_frame == "lidar_link"
    assert cfg.camera_frame == "camera_link"
    assert cfg.sensor_height == 0.30
    assert cfg.lidar_offset == (0.0, 0.0, 0.30)
    assert cfg.camera_offset == (0.0, 0.0, 0.30)
    assert cfg.camera_rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.robot_radius == 0.35
    assert cfg.robot_height == 0.30
    # Its fixes are baked into the derived USD, so no runtime stage patch is needed.
    assert cfg.stage_patch_fn is None
    assert cfg.articulation_cfg is not None
    assert cfg.action_cfg is not None
    assert cfg.action_cfg.wheel_radius == 0.1225
    assert cfg.action_cfg.wheel_base == 0.4523232
    assert cfg.action_cfg.left_wheel_joint_name == "left_wheel_joint"
    assert cfg.action_cfg.right_wheel_joint_name == "right_wheel_joint"


def test_custom_embodiment_registration():
    """Verify custom embodiment registration via instance and factory callable."""
    custom_cfg = RobotEmbodimentCfg(
        name="custom_bot",
        wheel_radius=0.10,
        wheel_base=0.30,
        robot_radius=0.25,
    )
    register_embodiment("custom_bot", custom_cfg)
    assert "custom_bot" in list_embodiments()
    retrieved = get_embodiment("custom_bot")
    assert retrieved.name == "custom_bot"
    assert retrieved.wheel_radius == 0.10

    # Test factory callable
    count = 0

    def factory():
        nonlocal count
        count += 1
        return RobotEmbodimentCfg(name=f"factory_bot_{count}", wheel_radius=0.05)

    register_embodiment("factory_bot", factory)
    b1 = get_embodiment("factory_bot")
    b2 = get_embodiment("factory_bot")
    assert b1.name == "factory_bot_1"
    assert b2.name == "factory_bot_2"
    assert b1 is not b2


def test_unknown_embodiment_raises_keyerror():
    """Verify requesting an unregistered embodiment raises KeyError with available list."""
    with pytest.raises(KeyError, match="Embodiment 'non_existent' not found in registry"):
        get_embodiment("non_existent")


def test_clear_registry_and_restore():
    """Verify clear_registry wipes registry and register_default_embodiments restores it."""
    clear_registry()
    assert list_embodiments() == []
    with pytest.raises(KeyError):
        get_embodiment("nova_carter")

    register_default_embodiments()
    assert "nova_carter" in list_embodiments()
    assert "dingo" in list_embodiments()


def test_register_embodiment_decorator():
    """Verify register_embodiment works as a class decorator."""
    @register_embodiment("decorated_bot")
    class DecoratedBotCfg(RobotEmbodimentCfg):
        name: str = "decorated_bot"
        wheel_radius: float = 0.08

    assert "decorated_bot" in list_embodiments()
    bot = get_embodiment("decorated_bot")
    assert isinstance(bot, DecoratedBotCfg)
    assert bot.name == "decorated_bot"
    assert bot.wheel_radius == 0.08


def test_embodiment_instance_isolation():
    """Verify modifying an instance retrieved via get_embodiment does not mutate registered instance."""
    base_instance = RobotEmbodimentCfg(
        name="isolation_bot",
        wheel_radius=0.15,
        sensor_height=0.30,
    )
    register_embodiment("isolation_bot", base_instance)

    b1 = get_embodiment("isolation_bot")
    b1.sensor_height = 0.99
    b1.wheel_radius = 0.99

    b2 = get_embodiment("isolation_bot")
    assert b2.sensor_height == 0.30
    assert b2.wheel_radius == 0.15


def test_embodiment_footprint_list_and_tuple():
    """Verify RobotEmbodimentCfg accepts both list and tuple footprint formats."""
    list_footprint = [(0.2, 0.2), (0.2, -0.2), (-0.2, -0.2), (-0.2, 0.2)]
    cfg1 = RobotEmbodimentCfg(name="list_fp", footprint=list_footprint)
    assert cfg1.footprint == list_footprint

    tuple_footprint = ((0.2, 0.2), (0.2, -0.2), (-0.2, -0.2), (-0.2, 0.2))
    cfg2 = RobotEmbodimentCfg(name="tuple_fp", footprint=tuple_footprint)
    assert cfg2.footprint == tuple_footprint


def test_zero_ros2_imports_runtime():
    """Verify in an isolated Python process that importing nav_arena.embodiments loads zero ROS 2 modules."""
    code = (
        "import sys, nav_arena.embodiments\n"
        "ros2_patterns = ('rclpy', 'ros2', 'builtin_interfaces', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'std_msgs', 'tf2_ros')\n"
        "loaded_ros2 = [m for m in sys.modules if any(m == p or m.startswith(f'{p}.') for p in ros2_patterns)]\n"
        "if loaded_ros2:\n"
        "    print(f'ERROR: {loaded_ros2}')\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"ROS 2 modules loaded when importing embodiments: {result.stdout} {result.stderr}"


def test_zero_ros2_imports_static_ast():
    """Verify through static AST analysis that embodiments source files contain zero ROS 2 imports."""
    embodiments_dir = Path(embodiments.__file__).resolve().parent
    py_files = list(embodiments_dir.glob("*.py"))
    assert len(py_files) >= 5, f"Expected at least 5 python files in {embodiments_dir}"

    forbidden_modules = {
        "rclpy",
        "ros2",
        "builtin_interfaces",
        "geometry_msgs",
        "nav_msgs",
        "sensor_msgs",
        "std_msgs",
        "tf2_ros",
    }

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    assert root_mod not in forbidden_modules, (
                        f"Forbidden ROS 2 import '{alias.name}' in {py_file.name}:{node.lineno}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    assert root_mod not in forbidden_modules, (
                        f"Forbidden ROS 2 import from '{node.module}' in {py_file.name}:{node.lineno}"
                    )


def test_urdf_generation_with_embodiments():
    """Verify generate_minimal_urdf works seamlessly with both Nova Carter and Dingo configs."""
    import xml.etree.ElementTree as ET

    # Nova Carter URDF
    carter_cfg = get_embodiment("nova_carter")
    carter_urdf = generate_minimal_urdf(carter_cfg)
    root_carter = ET.fromstring(carter_urdf)
    assert root_carter.attrib.get("name") == "nova_carter"

    # Dingo URDF
    dingo_cfg = get_embodiment("dingo")
    dingo_urdf = generate_minimal_urdf(dingo_cfg)
    root_dingo = ET.fromstring(dingo_urdf)
    assert root_dingo.attrib.get("name") == "dingo"


def test_wheeled_embodiments_are_differential_drive_without_strafing():
    """Verify the existing robots are differential drives with the historical timing and no lateral speed."""
    for name in ("nova_carter", "dingo"):
        cfg = get_embodiment(name)
        assert cfg.drive_type == "diff"
        assert cfg.max_lateral_speed == 0.0
        assert (cfg.sim_dt, cfg.decimation, cfg.spawn_height) == (0.01, 2, 0.25)
        assert cfg.step_dt == pytest.approx(0.02)


def test_unknown_drive_type_is_rejected():
    """Verify a typo in drive_type fails at construction instead of silently selecting the wrong behaviour."""
    with pytest.raises(ValueError, match="drive_type"):
        RobotEmbodimentCfg(drive_type="omni")


def test_policy_driven_drive_types():
    """Verify only the legged drive types are policy driven."""
    from nav_arena.embodiments.base import is_policy_driven

    assert is_policy_driven("quadruped")
    assert not is_policy_driven("diff") and not is_policy_driven("holonomic")


def test_contact_body_expr_defaults_to_body_link():
    """Verify contact sensing falls back to body_link unless contact_bodies is set."""
    cfg = RobotEmbodimentCfg(body_link="base_link")
    assert cfg.contact_body_expr == "base_link"
    assert RobotEmbodimentCfg(body_link="base", contact_bodies="(base|.*_calf)").contact_body_expr == "(base|.*_calf)"


# ==============================================================================
# Embodiment variants
# ==============================================================================


def _register_two_variant_robot(**kwargs):
    """Register a robot whose camera sits on a mast by default and at a low native pose in the other variant."""
    from nav_arena.embodiments import EmbodimentVariant

    base = RobotEmbodimentCfg(camera_offset=(0.0, 0.0, 0.30))
    register_embodiment(
        "variant_bot",
        base,
        variants=[
            EmbodimentVariant("mast", "Camera on a virtual mast.", {"camera_offset": (0.0, 0.0, 0.30)}),
            EmbodimentVariant("native", "Real sensor pose.", {"camera_offset": (0.05, 0.0, 0.10)}),
            EmbodimentVariant("blind", "No camera.", {"sensors": ("lidar",)}),
        ],
        default_variant=kwargs.get("default_variant", "mast"),
    )


def test_bare_name_resolves_to_the_default_variant_and_dotted_name_to_another():
    """Verify `robot` selects the default variant and `robot.variant` any other, each with its overrides applied."""
    _register_two_variant_robot()
    default = get_embodiment("variant_bot")
    assert (default.name, default.variant, default.full_name) == ("variant_bot", "mast", "variant_bot.mast")
    assert default.camera_offset == (0.0, 0.0, 0.30)
    native = get_embodiment("variant_bot.native")
    assert native.full_name == "variant_bot.native" and native.camera_offset == (0.05, 0.0, 0.10)
    assert get_embodiment("variant_bot.blind").sensors == ("lidar",)


def test_canonical_name_pins_the_default_variant():
    """Verify resolve_embodiment_name records the explicit variant, and leaves variantless robots alone."""
    from nav_arena.embodiments import resolve_embodiment_name

    _register_two_variant_robot()
    assert resolve_embodiment_name("variant_bot") == "variant_bot.mast"
    assert resolve_embodiment_name("variant_bot.native") == "variant_bot.native"
    assert resolve_embodiment_name("dingo") == "dingo"


@pytest.mark.parametrize("bad", ["variant_bot.nope", "dingo.mast", "missing_bot", "missing_bot.mast"])
def test_unknown_robot_or_variant_is_a_key_error(bad):
    """Verify unknown robots and variants fail with a KeyError naming what is available."""
    _register_two_variant_robot()
    with pytest.raises(KeyError):
        get_embodiment(bad)


def test_variant_registration_is_validated():
    """Verify typos in override fields, missing defaults, duplicates and dotted names fail at registration."""
    from nav_arena.embodiments import EmbodimentVariant

    base = RobotEmbodimentCfg()
    ok = EmbodimentVariant("a", "", {"camera_offset": (0.0, 0.0, 0.2)})
    with pytest.raises(ValueError, match="unknown or reserved fields"):
        register_embodiment("r", base, variants=[EmbodimentVariant("a", "", {"camera_ofset": (0, 0, 0)})], default_variant="a")
    with pytest.raises(ValueError, match="unknown or reserved fields"):
        register_embodiment("r", base, variants=[EmbodimentVariant("a", "", {"variant": "x"})], default_variant="a")
    with pytest.raises(ValueError, match="default_variant"):
        register_embodiment("r", base, variants=[ok])
    with pytest.raises(ValueError, match="default_variant"):
        register_embodiment("r", base, variants=[ok], default_variant="b")
    with pytest.raises(ValueError, match="duplicate"):
        register_embodiment("r", base, variants=[ok, ok], default_variant="a")
    with pytest.raises(ValueError, match="contain no"):
        register_embodiment("r", base, variants=[EmbodimentVariant("a.b")], default_variant="a.b")
    with pytest.raises(ValueError, match="must not contain"):
        register_embodiment("r.x", base, variants=[ok], default_variant="a")


def test_reregistering_without_variants_keeps_them_but_empty_tuple_clears_them():
    """Verify a module that registers itself on import cannot wipe the variants declared for its name."""
    from nav_arena.embodiments import list_variants

    _register_two_variant_robot()
    register_embodiment("variant_bot", RobotEmbodimentCfg())  # variants=None keeps them
    assert [v.name for v in list_variants("variant_bot")] == ["mast", "native", "blind"]
    register_embodiment("variant_bot", RobotEmbodimentCfg(), variants=())
    assert list_variants("variant_bot") == []
    assert get_embodiment("variant_bot").variant == ""


def test_lazy_registration_does_not_import_the_embodiment_until_needed():
    """Verify built-in robots are lazy strings (CLI-safe) and resolve to the same cfg when requested."""
    from nav_arena.embodiments import registry

    clear_registry()
    register_default_embodiments()
    assert isinstance(registry._EMBODIMENT_REGISTRY["nova_carter"], str)
    assert get_embodiment("nova_carter").wheel_radius == pytest.approx(0.14)


def test_registered_drive_type_matches_the_embodiment():
    """Verify the light drive_type recorded for the CLI listing agrees with each resolved embodiment."""
    from nav_arena.embodiments import drive_type_of

    for name in list_embodiments():
        assert drive_type_of(name) == get_embodiment(name).drive_type


def test_embodiments_package_and_registry_import_no_simulator_or_ml_modules():
    """Verify robot names, variants and resolution load without torch, Isaac Lab or Omniverse (the CLI relies on it)."""
    code = (
        "import sys\n"
        "from nav_arena.embodiments import list_embodiments, resolve_embodiment_name, list_variants\n"
        "list_embodiments(); resolve_embodiment_name('dingo')\n"
        "forbidden = {'isaacsim', 'isaaclab', 'omni', 'pxr', 'rclpy', 'torch'}\n"
        "loaded = forbidden.intersection(m.split('.')[0] for m in sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_unknown_sensor_names_are_rejected():
    """Verify a typo in the sensors tuple fails at construction."""
    with pytest.raises(ValueError, match="sensors"):
        RobotEmbodimentCfg(sensors=("lidar", "cam"))


def test_kaya_variants_keep_their_data_across_a_registry_reset():
    """Verify kaya.native keeps its camera after clear_registry + register_default_embodiments (a past bug)."""
    from nav_arena.embodiments.kaya_variants import MAST_CAMERA_OFFSET, NATIVE_CAMERA_OFFSET

    for _ in range(2):
        assert get_embodiment("kaya.native").camera_offset == NATIVE_CAMERA_OFFSET
        assert get_embodiment("kaya").camera_offset == MAST_CAMERA_OFFSET
        assert get_embodiment("kaya").full_name == "kaya.mast"
        clear_registry()
        register_default_embodiments()


def test_kaya_lidar_and_mast_camera_sit_0_30_m_above_the_floor():
    """Verify Kaya's LiDAR and mast camera share the 0.30 m mount height (offsets are from base_link, 0.091 m up)."""
    from nav_arena.embodiments.kaya_variants import BASE_LINK_HEIGHT

    kaya = get_embodiment("kaya")
    assert kaya.lidar_offset[2] + BASE_LINK_HEIGHT == pytest.approx(0.30)
    assert kaya.camera_offset[2] + BASE_LINK_HEIGHT == pytest.approx(0.30)
    assert kaya.sensor_height == kaya.lidar_offset[2]
