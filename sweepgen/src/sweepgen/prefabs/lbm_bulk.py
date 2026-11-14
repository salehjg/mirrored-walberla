# This file is part of waLBerla. waLBerla is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# waLBerla is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.

import copy

from pystencils import fields, CreateKernelConfig, Field

from pystencilssfg import SfgComposer, SfgContext
from pystencilssfg.composer import SfgIComposer
from pystencilssfg.composer.custom import CustomGenerator
from lbmpy import LBMConfig, LBMOptimisation, create_lb_update_rule
from lbmpy.advanced_streaming import Timestep
from lbmpy.methods import LbmCollisionRule, AbstractLbMethod
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter

from ..build_config import get_build_config
from ..sweep import Sweep


class LbmBulk(CustomGenerator):
    """Generate a sweep group for lattice Boltzmann bulk dynamics.

    Generate a sweep group for a given lattice Boltzmann method definition,
    including:

    - A PDF initialization sweep
    - A stream/collide sweep: ``StreamCollide`` for the ``pull`` streaming pattern (two PDF
      fields, swapped after every sweep), or ``StreamCollideEven`` and ``StreamCollideOdd`` for
      the in-place ``aa`` pattern (one PDF field; even and odd time steps alternate, starting
      with an even step on the PDFs written by ``InitPdfs``)

    With the ``aa`` pattern, the ghost layers must be exchanged with
    ``halo_exchange::aaPullSync`` after every even step and with
    ``halo_exchange::aaPushSync`` after every odd step.

    With ``output_free_sweeps=True``, the group also contains stream/collide sweeps that do not
    write the density and velocity fields (``StreamCollideNoOutput``, or
    ``StreamCollideEvenNoOutput`` and ``StreamCollideOddNoOutput``). They update the PDFs exactly
    like their counterparts and move fewer bytes per cell (4 fewer store streams in 3D); use them
    for the time steps whose macroscopic values are not needed.

    **Example:**

    .. code-block:: python

        lb_config = LBMConfig(
            stencil=Stencil.D3Q19,
            method=Method.CENTRAL_MOMENT,
            relaxation_rate=sp.Symbol("omega")
        )
        lb_bulk = LbmBulk(sfg, "MyLBM", lb_config)
        sfg.generate(lb_bulk)

    .. admonition:: Example Apps
        :class: seealso

        - :walberla-example-app:`DoubleShearLayer`
        - :walberla-example-app:`ParallelPlates`

    Args:
        sfg: An SFG composer or context that holds a waLBerla build configuration
        name: Name of the sweep group
        lbm_config: lbmpy method configuration defining the lattice Boltzmann method
        lbm_optimisation: Optimization settings for lbmpy
        gen_config: pystencils code generator configuration
        output_free_sweeps: Also generate the stream/collide sweeps without density/velocity output
    """

    def __init__(
        self,
        sfg: SfgIComposer | SfgContext,
        name: str,
        lbm_config: LBMConfig | None = None,
        lbm_optimisation: LBMOptimisation | None = None,
        gen_config: CreateKernelConfig | None = None,
        output_free_sweeps: bool = False,
    ):
        self._name = name
        self._output_free_sweeps = output_free_sweeps
        if lbm_config is None:
            lbm_config = LBMConfig()

        if lbm_config.streaming_pattern not in ("pull", "aa"):
            raise ValueError("Only the `pull` and `aa` streaming patterns are currently supported.")
        self._in_place = lbm_config.streaming_pattern == "aa"

        self._lbm_config = lbm_config

        base_gen_config = get_build_config(sfg).get_pystencils_config()
        if gen_config is not None:
            base_gen_config.override(gen_config)
        self._gen_config = base_gen_config

        self._stencil = lbm_config.stencil
        Q, D = self._stencil.Q, self._stencil.D

        self._pdfs: Field
        self._pdfs_tmp: Field | None
        self._rho: Field
        self._u: Field

        dtype = self._gen_config.get_option("default_dtype")

        if lbm_optimisation is None:
            lbm_optimisation = LBMOptimisation(
                simplification="auto",
                cse_global=False,
                field_layout="fzyx",
            )

        if lbm_optimisation.symbolic_field is None:
            lbm_optimisation.symbolic_field = Field.create_generic(
                "f",
                spatial_dimensions=D,
                index_shape=(Q,),
                layout=lbm_optimisation.field_layout,
                dtype=dtype,
            )

        self._pdfs = lbm_optimisation.symbolic_field

        if self._in_place:
            #   In-place streaming: lbmpy reads and writes the one PDF field only
            self._pdfs_tmp = None
        else:
            if lbm_optimisation.symbolic_temporary_field is None:
                lbm_optimisation.symbolic_temporary_field = (
                    Field.create_generic(
                        "f_tmp",
                        spatial_dimensions=D,
                        index_shape=(Q,),
                        layout=lbm_optimisation.field_layout,
                        dtype=dtype,
                    )
                )

            self._pdfs_tmp = lbm_optimisation.symbolic_temporary_field

        self._rho, self._u = fields(  # type: ignore
            f"rho, u({D}): {dtype}[{D}D]", layout=lbm_optimisation.field_layout
        )

        output_fields = dict({"density": self._rho, "velocity": self._u})

        def update_rule(timestep: Timestep, output: bool = True) -> LbmCollisionRule:
            cfg = copy.copy(lbm_config)
            cfg.timestep = timestep
            return create_lb_update_rule(
                lbm_config=cfg, lbm_optimisation=lbm_optimisation, output=output_fields if output else {}
            )

        self._update_rule: LbmCollisionRule
        self._update_rule_odd: LbmCollisionRule | None
        if self._in_place:
            self._update_rule = update_rule(Timestep.EVEN)
            self._update_rule_odd = update_rule(Timestep.ODD)
        else:
            self._update_rule = update_rule(lbm_config.timestep)
            self._update_rule_odd = None

        #   (sweep name, update rule) of the sweeps without density/velocity output
        self._output_free: list[tuple[str, LbmCollisionRule]] = []
        if output_free_sweeps:
            if self._in_place:
                self._output_free = [
                    ("StreamCollideEvenNoOutput", update_rule(Timestep.EVEN, output=False)),
                    ("StreamCollideOddNoOutput", update_rule(Timestep.ODD, output=False)),
                ]
            else:
                self._output_free = [("StreamCollideNoOutput", update_rule(lbm_config.timestep, output=False))]

        #   Equilibrium PDFs where the first stream/collide sweep reads its pre-collision values.
        #   For `aa` that is the even step's own cell; for `pull` the setter writes slot i of
        #   cell x - c_i, i.e. into the neighbors.
        init_streaming = (
            dict(streaming_pattern="aa", previous_timestep=Timestep.EVEN) if self._in_place else dict()
        )
        self._init_rule = macroscopic_values_setter(
            lb_method=self._update_rule.method,
            density=self._rho.center,
            velocity=self._u.center_vector,
            pdfs=self._pdfs,
            set_pre_collision_pdfs=True,
            **init_streaming,
        )

    @property
    def pdfs(self) -> Field:
        """The symbolic PDF field"""
        return self._pdfs

    @property
    def rho(self) -> Field:
        """The symbolic density field"""
        return self._rho

    @property
    def u(self) -> Field:
        """The symbolic velocity field"""
        return self._u

    @property
    def update_rule(self) -> LbmCollisionRule:
        """The symbolic LB update rule (of the even time step for the ``aa`` pattern)"""
        return self._update_rule

    @property
    def update_rule_odd(self) -> LbmCollisionRule | None:
        """The symbolic LB update rule of the odd time step (``aa`` pattern only)"""
        return self._update_rule_odd

    @property
    def lb_method(self) -> AbstractLbMethod:
        """The LB method object"""
        return self._update_rule.method

    def generate(self, sfg: SfgComposer) -> None:
        with sfg.namespace(self._name):
            if self._in_place:
                assert self._update_rule_odd is not None
                sfg.generate(
                    Sweep("StreamCollideEven", self._update_rule, self._gen_config)
                )
                sfg.generate(
                    Sweep("StreamCollideOdd", self._update_rule_odd, self._gen_config)
                )
            else:
                assert self._pdfs_tmp is not None
                stream_collide_sweep = Sweep(
                    "StreamCollide", self._update_rule, self._gen_config
                )
                stream_collide_sweep.swap_fields(self._pdfs, self._pdfs_tmp)
                sfg.generate(stream_collide_sweep)

            for sweep_name, rule in self._output_free:
                sweep = Sweep(sweep_name, rule, self._gen_config)
                if not self._in_place:
                    assert self._pdfs_tmp is not None
                    sweep.swap_fields(self._pdfs, self._pdfs_tmp)
                sfg.generate(sweep)

            init_sweep = Sweep("InitPdfs", self._init_rule, self._gen_config)
            sfg.generate(init_sweep)
