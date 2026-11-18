#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the support polygon limit."""

import unittest

import numpy as np
import pinocchio as pin

from pink import Configuration, solve_ik
from pink.exceptions import FrameNotFound, LimitDefinitionError
from pink.limits import SupportPolygonLimit
from pink.tasks import ComTask

# A square support polygon, 0.4 m wide, centered on the frame's origin.
_SQUARE = np.array([[0.2, 0.2], [-0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]])


def _link(name: str, mass: float, com=(0.0, 0.0, 0.0)) -> str:
    """URDF link with a point-like inertia of the given mass at ``com``."""
    return f"""
  <link name="{name}">
    <inertial>
      <origin xyz="{com[0]} {com[1]} {com[2]}"/>
      <mass value="{mass}"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>"""


def _revolute(name: str, parent: str, child: str, xyz: str, axis: str) -> str:
    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz}"/>
    <axis xyz="{axis}"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>"""


# A planar mobile base (x, y, yaw) carrying a two-link arm that pitches. With
# the arm upright, the CoM lies above the base's origin.
_MOBILE_URDF = (
    '<robot name="mobile">'
    + _link("base", 10.0)
    + _link("link1", 2.0, (0, 0, 0.25))
    + _link("link2", 1.0, (0, 0, 0.25))
    + _link("support", 0.0)
    + """
  <joint name="planar" type="planar">
    <parent link="world_link"/>
    <child link="base"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="support_joint" type="fixed">
    <parent link="base"/>
    <child link="support"/>
  </joint>"""
    + _revolute("j1", "base", "link1", "0 0 0.05", "0 1 0")
    + _revolute("j2", "link1", "link2", "0 0 0.5", "0 1 0")
    + "</robot>"
).replace(
    '<link name="base">',
    '<link name="world_link"/><link name="base">',
    1,
)

# A floating torso with a swinging arm; its polygon is given in the world.
_FLOATING_URDF = (
    '<robot name="floating">'
    + _link("torso", 5.0)
    + _link("arm", 1.0, (0.2, 0, 0))
    + _revolute("shoulder", "torso", "arm", "0 0.1 0.1", "0 1 0")
    + "</robot>"
)

# A cart on a rail: the CoM's x is q, linear in q. (Pinocchio does not count
# the root link's inertia in the CoM, so the fixed body is left massless.)
_CART_URDF = (
    '<robot name="cart">'
    + '<link name="robot"/>'
    + _link("cart", 1.0)
    + """
  <joint name="rail" type="prismatic">
    <parent link="robot"/>
    <child link="cart"/>
    <axis xyz="1 0 0"/>
    <limit lower="-10" upper="10" effort="100" velocity="10"/>
  </joint>
</robot>"""
)


class TestSupportPolygonLimit(unittest.TestCase):
    """Test the support polygon limit."""

    @classmethod
    def setUpClass(cls):
        """Build the inline models."""
        cls.mobile = pin.buildModelFromXML(_MOBILE_URDF)
        cls.floating = pin.buildModelFromXML(
            _FLOATING_URDF, pin.JointModelFreeFlyer()
        )
        cls.cart = pin.buildModelFromXML(_CART_URDF)

    @staticmethod
    def _configuration(model: pin.Model, q: np.ndarray) -> Configuration:
        return Configuration(model, model.createData(), q)

    def _mobile_q(self, x, y, yaw, j1, j2) -> np.ndarray:
        return np.array([x, y, np.cos(yaw), np.sin(yaw), j1, j2])

    def _mobile_limit(self, **kwargs) -> SupportPolygonLimit:
        return SupportPolygonLimit.from_vertices(
            self.mobile, "support", _SQUARE, **kwargs
        )

    def test_mobile_model_layout(self):
        """Sanity check of the inline model: a planar base, two arm joints."""
        self.assertEqual((self.mobile.nq, self.mobile.nv), (6, 5))

    def test_invalid_definitions_raise(self):
        """Ill-formed polygons, margins, gains or models raise."""
        A, b = np.eye(2), np.ones(2)
        model = self.mobile
        bad = [
            lambda: SupportPolygonLimit(model, "support", np.ones((2, 3)), b),
            lambda: SupportPolygonLimit(model, "support", np.ones((3, 0)), b),
            lambda: SupportPolygonLimit(model, "support", A, np.ones(3)),
            lambda: SupportPolygonLimit(
                model, "support", [[np.nan, 0.0], [0.0, 1.0]], b
            ),
            lambda: SupportPolygonLimit(model, "support", A, [1.0, np.inf]),
            lambda: SupportPolygonLimit(
                model, "support", [[0.0, 0.0], [0.0, 1.0]], b
            ),
            lambda: SupportPolygonLimit(model, "support", A, b, margin=np.inf),
            lambda: SupportPolygonLimit(model, "support", A, b, gain=0.0),
            lambda: SupportPolygonLimit(model, "support", A, b, gain=1.5),
            # A model without mass.
            lambda: SupportPolygonLimit(pin.Model(), "universe", A, b),
            # Mass only in the root link, which Pinocchio's CoM leaves out.
            lambda: SupportPolygonLimit(
                pin.buildModelFromXML(
                    '<robot name="r">' + _link("root", 1.0) + "</robot>"
                ),
                "universe",
                A,
                b,
            ),
        ]
        for i, make in enumerate(bad):
            with self.assertRaises(LimitDefinitionError, msg=f"case {i}"):
                make()
        with self.assertRaises(FrameNotFound):
            SupportPolygonLimit(self.mobile, "missing", A, b)

    def test_from_vertices_matches_polygon(self):
        """The convex hull of the vertices is the polygon, in any order."""
        rng = np.random.default_rng(0)
        points = rng.uniform(-0.3, 0.3, (200, 2))
        inside = np.all(np.abs(points) <= 0.2, axis=1)
        # Clockwise, counter-clockwise, and with an interior and repeated
        # vertex.
        for vertices in (
            _SQUARE,
            _SQUARE[::-1],
            np.vstack([_SQUARE, [0, 0]] * 2),
        ):
            limit = SupportPolygonLimit.from_vertices(
                self.mobile, "support", vertices
            )
            self.assertEqual(limit.A.shape, (4, 2))
            np.testing.assert_allclose(np.linalg.norm(limit.A, axis=1), 1.0)
            np.testing.assert_array_equal(
                np.all(points @ limit.A.T <= limit.b, axis=1), inside
            )

    def test_from_vertices_rejects_degenerate_polygons(self):
        """Fewer than 3 non-collinear vertices raise."""
        for vertices in (
            [[0, 0], [1, 0]],
            [[0, 0], [1, 0], [2, 0]],
            [[0, 0, 0]],
        ):
            with self.assertRaises(LimitDefinitionError):
                SupportPolygonLimit.from_vertices(
                    self.mobile, "support", vertices
                )

    def test_empty_polygon_is_inactive(self):
        """An empty polygon adds no constraint."""
        limit = SupportPolygonLimit(
            self.mobile, "support", np.zeros((0, 2)), np.zeros(0)
        )
        configuration = self._configuration(
            self.mobile, pin.neutral(self.mobile)
        )
        self.assertIsNone(limit.compute_qp_inequalities(configuration, 0.1))
        limit = SupportPolygonLimit(self.mobile, "support", [], [])
        self.assertIsNone(limit.compute_qp_inequalities(configuration, 0.1))

    def test_rows_normalized(self):
        """Half-space rows are scaled to unit normals."""
        limit = SupportPolygonLimit(
            self.mobile, "support", [[2.0, 0.0], [0.0, -3.0]], [0.4, 0.6]
        )
        np.testing.assert_allclose(limit.A, [[1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(limit.b, [0.2, 0.2])

    def test_distances_follow_the_base(self):
        """Distances follow the base.

        With the arm upright the CoM is above the base's origin, wherever the
        base is: every edge is 0.2 m away.
        """
        limit = self._mobile_limit(margin=0.05, gain=0.5)
        for base in ([0.0, 0.0, 0.0], [1.0, -2.0, 0.7]):
            configuration = self._configuration(
                self.mobile, self._mobile_q(*base, 0.0, 0.0)
            )
            np.testing.assert_allclose(
                limit.compute_distances(configuration), 0.2
            )
            result = limit.compute_qp_inequalities(configuration, 0.1)
            assert result is not None
            np.testing.assert_allclose(result[1], 0.5 * (0.2 - 0.05))

    def test_distances_in_a_tilted_offset_frame(self):
        """Distances, h and G are right in a tilted, offset frame.

        The frame is tilted and offset from its joint, the polygon is
        asymmetric, and the reference is an independent computation.
        """
        model = pin.Model(self.mobile)
        j1 = model.getJointId("j1")
        placement = pin.SE3(
            pin.rpy.rpyToMatrix(0.3, -0.2, 0.7), np.array([0.1, -0.05, 0.2])
        )
        model.addFrame(
            pin.Frame(
                "tilted",
                j1,
                model.getFrameId("link1"),
                placement,
                pin.FrameType.OP_FRAME,
            )
        )
        A = np.array([[1.0, 0.5], [-0.3, 1.0], [-1.0, -2.0]])
        b = np.array([0.5, 0.4, 0.6])
        limit = SupportPolygonLimit(
            model, "tilted", A, b, margin=0.02, gain=0.7
        )
        q0 = self._mobile_q(0.3, -0.1, 0.5, 0.4, -0.7)
        configuration = self._configuration(model, q0)

        data = model.createData()
        com = pin.centerOfMass(model, data, q0)
        pin.framesForwardKinematics(model, data, q0)
        oMf = data.oMf[model.getFrameId("tilted")]
        com_F = oMf.rotation.T @ (com - oMf.translation)
        norms = np.linalg.norm(A, axis=1)
        expected = (b - A @ com_F[:2]) / norms
        np.testing.assert_allclose(
            limit.compute_distances(configuration), expected, atol=1e-12
        )
        result = limit.compute_qp_inequalities(configuration, 0.1)
        assert result is not None
        G, h = result
        np.testing.assert_allclose(h, 0.7 * (expected - 0.02), atol=1e-12)

        # The arm joint carrying the frame moves it, so the planar base is
        # the only part that drops out.
        np.testing.assert_allclose(G[:, :3], 0.0, atol=1e-12)
        self.assertGreater(np.abs(G[:, 3:]).max(), 0.01)
        eps = 1e-6
        dq = np.random.default_rng(2).uniform(-1.0, 1.0, model.nv)
        d = []
        for sign in (1.0, -1.0):
            configuration.update(pin.integrate(model, q0, sign * eps * dq))
            d.append(limit.compute_distances(configuration))
        np.testing.assert_allclose(
            (d[0] - d[1]) / (2 * eps), -G @ dq, atol=1e-8
        )

    def test_jacobian_matches_finite_differences(self):
        """G dq is the first-order decrease of the distances along dq.

        Checked for a frame on a planar base and for the world frame under a
        floating base.
        """
        rng = np.random.default_rng(1)
        q_floating = pin.neutral(self.floating)
        q_floating[:3] = [0.1, -0.2, 1.0]
        q_floating[3:7] = [0.1, 0.3, -0.2, 0.9]
        q_floating[3:7] /= np.linalg.norm(q_floating[3:7])
        q_floating[7] = 0.4
        cases = [
            (
                self._mobile_limit(),
                self.mobile,
                self._mobile_q(0.3, -0.1, 0.5, 0.4, -0.7),
            ),
            (
                SupportPolygonLimit.from_vertices(
                    self.floating, "universe", _SQUARE
                ),
                self.floating,
                q_floating,
            ),
        ]
        eps = 1e-6
        for limit, model, q0 in cases:
            configuration = self._configuration(model, q0)
            result = limit.compute_qp_inequalities(configuration, 0.1)
            assert result is not None
            G = result[0]
            for _ in range(3):
                dq = rng.uniform(-1.0, 1.0, model.nv)
                d = []
                for sign in (1.0, -1.0):
                    configuration.update(
                        pin.integrate(model, q0, sign * eps * dq)
                    )
                    d.append(limit.compute_distances(configuration))
                np.testing.assert_allclose(
                    (d[0] - d[1]) / (2 * eps), -G @ dq, atol=1e-8
                )

    def test_planar_base_does_not_enter(self):
        """A frame on a planar base zeroes the base's columns of G."""
        limit = self._mobile_limit()
        configuration = self._configuration(
            self.mobile, self._mobile_q(0.3, -0.1, 0.5, 0.4, -0.7)
        )
        result = limit.compute_qp_inequalities(configuration, 0.1)
        assert result is not None
        G = result[0]
        np.testing.assert_allclose(G[:, :3], 0.0, atol=1e-12)
        self.assertGreater(np.abs(G[:, 3:]).max(), 0.01)

    def test_clamp_violated(self):
        """Clamping violated rows changes h only."""
        configuration = self._configuration(
            self.mobile, self._mobile_q(0.0, 0.0, 0.0, 0.0, 0.0)
        )
        # A margin of 0.3 m puts the CoM 0.1 m inside the band of every edge.
        retreat = self._mobile_limit(margin=0.3).compute_qp_inequalities(
            configuration, 0.1
        )
        hold = self._mobile_limit(
            margin=0.3, clamp_violated=True
        ).compute_qp_inequalities(configuration, 0.1)
        assert retreat is not None and hold is not None
        np.testing.assert_allclose(retreat[1], -0.1)
        np.testing.assert_array_equal(hold[1], 0.0)
        np.testing.assert_array_equal(retreat[0], hold[0])

    def _cart_step(self, q: float, target: float, **kwargs) -> float:
        """Take one IK step of a CoM task towards ``target``.

        Returns:
            The CoM's x after the step.
        """
        configuration = self._configuration(self.cart, np.array([q]))
        limit = SupportPolygonLimit.from_vertices(
            self.cart, "universe", _SQUARE, margin=0.05, **kwargs
        )
        task = ComTask(cost=1.0)
        task.set_target(np.array([target, 0.0, 0.0]))
        velocity = solve_ik(
            configuration, [task], 1.0, solver="daqp", limits=[limit]
        )
        configuration.integrate_inplace(velocity, 1.0)
        return float(pin.centerOfMass(self.cart, configuration.data)[0])

    def test_solve_ik_stops_at_the_margin(self):
        """The CoM is linear in q here, so the limit holds exactly."""
        self.assertAlmostEqual(self._cart_step(0.0, 1.0), 0.15, places=6)
        # With gain 1/2, the CoM may close half of its 0.15 m to the margin.
        self.assertAlmostEqual(
            self._cart_step(0.0, 1.0, gain=0.5), 0.075, places=6
        )

    def test_solve_ik_retreats_or_holds(self):
        """A violated start retreats, or holds with clamp_violated.

        From a CoM 0.1 m beyond the margin (x = 0.25), the default limit
        moves it back to the margin; with clamp_violated it may stay.
        """
        self.assertAlmostEqual(self._cart_step(0.25, 0.25), 0.15, places=6)
        self.assertAlmostEqual(
            self._cart_step(0.25, 0.25, clamp_violated=True), 0.25, places=6
        )
        self.assertLessEqual(
            self._cart_step(0.25, 1.0, clamp_violated=True), 0.25 + 1e-9
        )


if __name__ == "__main__":
    unittest.main()
