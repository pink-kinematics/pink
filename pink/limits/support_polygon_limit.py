#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Support polygon (static stability) limit."""

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ..exceptions import FrameNotFound, LimitDefinitionError
from .limit import Limit

if TYPE_CHECKING:
    from ..configuration import Configuration


def _convex_hull_halfspaces(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the half-spaces ``A p <= b`` of the convex hull of 2D points.

    Uses Andrew's monotone chain. Each row of ``A`` is a unit outward normal,
    so ``b - A p`` is the distance from ``p`` to each edge's line.

    Args:
        vertices: Points, shape (n, 2).

    Returns:
        Pair ``(A, b)`` of normals and offsets.

    Raises:
        LimitDefinitionError: If the points do not span a polygon.
    """
    pts = np.unique(vertices, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float(
            (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])  # Counter-clockwise.
    if len(hull) < 3:
        raise LimitDefinitionError(
            "SupportPolygonLimit vertices must contain at least 3 "
            "non-collinear points"
        )

    edges = np.roll(hull, -1, axis=0) - hull
    # For a counter-clockwise polygon, (dy, -dx) is the outward normal.
    A = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    b = np.einsum("ij,ij->i", A, hull)
    return A, b


class SupportPolygonLimit(Limit):
    r"""Keep the center of mass (CoM) above a support polygon.

    A robot standing on the ground is statically stable when the projection of
    its CoM along gravity lies inside its support polygon, the convex hull of
    its ground contacts. This limit keeps the CoM :math:`c(q)` of the whole
    model at least a margin :math:`m` inside a convex polygon
    :math:`\{p : A p \leq b\}` given in the xy plane of a frame :math:`F`.

    Let :math:`p_F(q)` and :math:`R_F(q)` be the position and orientation of
    :math:`F` in the world, and let :math:`a_i^T p \leq b_i` be the polygon's
    half-spaces with unit normals :math:`a_i`. The CoM in :math:`F` is
    :math:`c_F = R_F^T (c - p_F)`, and its distance inside edge :math:`i` is
    :math:`d_i(q) = b_i - a_i^T S c_F`, where :math:`S` selects the x and y
    coordinates. The limit asks that :math:`d_i(q + \Delta q) \geq m`,
    linearized:

    .. math::

        A S J_{c/F}(q) \Delta q \leq \gamma \, (d(q) - m)

    where :math:`\gamma \in (0, 1]` is the gain and :math:`J_{c/F}` is the
    Jacobian of :math:`c_F`, the CoM velocity relative to :math:`F` expressed
    in :math:`F`:

    .. math::

        J_{c/F} = R_F^T \left( J_c - J_{F}(c) \right)

    with :math:`J_c` the CoM Jacobian and :math:`J_F(c)` the translational
    Jacobian of the point fixed to :math:`F` that coincides with the CoM. If
    :math:`F` is attached to the robot's base, a motion of the base moves the
    CoM and the polygon together, so the base's degrees of freedom do not
    enter the constraint.

    The formulation matches the ``StaticStabilityConstraint`` ingredient of
    Drake's ``DifferentialInverseKinematicsSystem``.

    .. note::

        The polygon is projected along :math:`F`'s z axis. Choose a frame
        whose z axis points against gravity, e.g. a frame on a wheeled base
        that drives on flat ground, or ``"universe"`` for a floating-base
        robot, with the polygon given in world coordinates.

    .. note::

        By default a row that is already violated (:math:`d_i < m`) asks the
        CoM to move back inside, by a fraction :math:`\gamma` of the
        shortfall per step, as joint position limits do, and the QP is
        infeasible if it cannot. Set ``clamp_violated`` to only
        forbid such a row from getting worse
        (:math:`h_i = \max(\gamma (d_i - m), 0)`); then :math:`\Delta q = 0`
        always satisfies the limit. A margin larger than the radius of the
        polygon's largest inscribed circle cannot be satisfied.

    .. note::

        The CoM is Pinocchio's (``pin.centerOfMass``), as in
        :class:`pink.tasks.ComTask`: it leaves out the inertia attached to
        the universe, such as the root link of a fixed-base model.

    Attributes:
        model: Pinocchio model.
        frame: Name of the frame :math:`F` the polygon is given in.
        frame_id: Index of the frame :math:`F` in the model.
        A: Unit outward normals of the polygon's edges, shape (k, 2).
        b: Offsets of the polygon's edges, shape (k,), so that ``b - A p`` is
            the distance of ``p`` inside each edge.
        margin: Distance :math:`m` in meters to keep inside every edge.
        gain: Gain :math:`\gamma` in (0, 1].
        clamp_violated: If True, rows that are already violated are only kept
            from getting worse.

    Example:

    .. code-block:: python

        # A square support polygon, 0.4 m wide, around the "base" frame.
        limit = SupportPolygonLimit.from_vertices(
            model,
            frame="base",
            vertices=[[0.2, 0.2], [-0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]],
            margin=0.02,
        )
        # A limits list replaces solve_ik's defaults, so keep them.
        limits = [
            model.configuration_limit,
            model.velocity_limit,
            limit,
        ]
        velocity = solve_ik(configuration, tasks, dt, "daqp", limits=limits)
    """

    model: pin.Model
    frame: str
    frame_id: int
    A: np.ndarray
    b: np.ndarray
    margin: float
    gain: float
    clamp_violated: bool

    def __init__(
        self,
        model: pin.Model,
        frame: str,
        A: npt.ArrayLike,
        b: npt.ArrayLike,
        margin: float = 0.0,
        gain: float = 1.0,
        clamp_violated: bool = False,
    ):
        r"""Initialize the support polygon limit.

        Args:
            model: Pinocchio model.
            frame: Name of the frame :math:`F` the polygon is given in. Any
                frame of the model works, including ``"universe"``.
            A: Normals of the polygon's half-spaces :math:`A p \leq b` in
                :math:`F`'s xy plane, shape (k, 2). Rows need not be
                normalized, but must be nonzero. An empty polygon (k = 0)
                adds no constraint.
            b: Offsets of the half-spaces, shape (k,).
            margin: Distance in meters to keep the CoM inside every edge. A
                negative value allows the CoM up to that far outside.
            gain: Fraction in (0, 1] of the distance to the margin that the
                CoM may close in a single step. Smaller values are safer but
                slow the approach.
            clamp_violated: If True, a row that is already violated is only
                kept from getting worse, so the limit never makes the QP
                infeasible. If False (default), it asks the CoM back inside.

        Raises:
            FrameNotFound: If the frame is not in the model.
            LimitDefinitionError: If the polygon, margin or gain is invalid,
                or the model has no mass.
        """
        name = self.__class__.__name__
        A = np.asarray(A, dtype=np.float64)
        b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        # An empty polygon is A of shape (0, 2), or (0,) as in A=[].
        if A.shape in ((0,), (0, 2)):
            A = A.reshape(0, 2)
        else:
            A = np.atleast_2d(A)
        if A.ndim != 2 or A.shape[1] != 2:
            raise LimitDefinitionError(
                f"{name} A must have shape (k, 2), got {A.shape}"
            )
        if b.shape != (A.shape[0],):
            raise LimitDefinitionError(
                f"{name} b must have shape ({A.shape[0]},), got {b.shape}"
            )
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
            raise LimitDefinitionError(f"{name} A and b must be finite")
        row_norms = np.linalg.norm(A, axis=1)
        if np.any(row_norms == 0.0):
            raise LimitDefinitionError(f"{name} A must not have zero rows")
        if not np.isfinite(margin):
            raise LimitDefinitionError(f"{name} margin must be finite")
        if not 0.0 < gain <= 1.0:
            raise LimitDefinitionError(
                f"{name} gain must be in the range (0, 1]"
            )
        if pin.computeTotalMass(model) <= 0.0:
            raise LimitDefinitionError(f"{name} model has no mass")
        if not model.existFrame(frame):
            raise FrameNotFound(frame, model.frames)

        self.model = model
        self.frame = frame
        self.frame_id = model.getFrameId(frame)
        self.A = A / row_norms[:, None]
        self.b = b / row_norms
        self.margin = float(margin)
        self.gain = float(gain)
        self.clamp_violated = clamp_violated

    @classmethod
    def from_vertices(
        cls,
        model: pin.Model,
        frame: str,
        vertices: npt.ArrayLike,
        **kwargs,
    ) -> "SupportPolygonLimit":
        """Create the limit from the vertices of the support polygon.

        Args:
            model: Pinocchio model.
            frame: Name of the frame the vertices are given in.
            vertices: Points in the frame's xy plane, shape (n, 2), in any
                order. The polygon is their convex hull, which needs at least
                3 non-collinear points.
            kwargs: Remaining arguments of :class:`SupportPolygonLimit`.

        Returns:
            The support polygon limit.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise LimitDefinitionError(
                f"{cls.__name__} vertices must have shape (n, 2), "
                f"got {vertices.shape}"
            )
        if not np.all(np.isfinite(vertices)):
            raise LimitDefinitionError(
                f"{cls.__name__} vertices must be finite"
            )
        A, b = _convex_hull_halfspaces(vertices)
        return cls(model, frame, A, b, **kwargs)

    def compute_distances(self, configuration: "Configuration") -> np.ndarray:
        """Compute the distance of the CoM's projection inside every edge.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Distances :math:`d(q)` in meters, shape (k,). Negative entries are
            edges the CoM lies outside of.
        """
        com = pin.centerOfMass(
            self.model, configuration.data, configuration.q, False
        )
        return self._distances(configuration, com)

    def _distances(
        self, configuration: "Configuration", com: np.ndarray
    ) -> np.ndarray:
        """Compute distances for a given world-frame CoM."""
        placement = configuration.data.oMf[self.frame_id]
        com_F = placement.rotation.T @ (com - placement.translation)
        return self.b - self.A @ com_F[:2]

    def compute_qp_inequalities(
        self,
        configuration: "Configuration",
        dt: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r"""Compute the support polygon limit as linearized QP inequalities.

        Args:
            configuration: Robot configuration :math:`q`.
            dt: Integration timestep in [s]. Unused: the limit bounds a
                displacement.

        Returns:
            Pair :math:`(G, h)` with :math:`G = A S J_{c/F}` and
            :math:`h = \gamma (d - m)`, or with :math:`h` clamped at zero if
            ``clamp_violated`` is set. ``None`` for an empty polygon.
        """
        del dt  # Unused.
        if self.A.shape[0] == 0:
            return None

        model = self.model
        data = configuration.data
        # Reshape: Pinocchio returns 1D arrays when nv = 1.
        jac_com = pin.jacobianCenterOfMass(
            model, data, configuration.q, False
        ).reshape(3, model.nv)
        com = np.array(data.com[0])
        placement = data.oMf[self.frame_id]
        jac_frame = pin.getFrameJacobian(
            model,
            data,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        ).reshape(6, model.nv)
        jac_point = (
            jac_frame[:3]
            - pin.skew(com - placement.translation) @ (jac_frame[3:])
        )
        jac_rel = placement.rotation.T @ (jac_com - jac_point)

        G = self.A @ jac_rel[:2]
        h = self.gain * (self._distances(configuration, com) - self.margin)
        if self.clamp_violated:
            h = np.maximum(h, 0.0)
        return G, h
