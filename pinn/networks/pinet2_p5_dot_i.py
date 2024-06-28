# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from pinn.layers import (
    CellListNL,
    CutoffFunc,
    PolynomialBasis,
    GaussianBasis,
    AtomicOnehot,
    ANNOutput,
)

from pinn.networks.pinet import FFLayer, PILayer, IPLayer, ResUpdate
from pinn.networks.pinet2 import ScaleLayer, OutLayer, DotLayer


class PIXLayer(tf.keras.layers.Layer):
    R"""`PIXLayer` takes the equalvariant properties ${}^{3}\mathbb{P}_{ix\zeta}$ as input and outputs interactions for each pair ${}^{3}\mathbb{I}_{ijx\zeta}$. The `PIXLayer` has two styles, specified by the `weighted` argument:

    `weighted`:
    $$
    \begin{aligned}
    {}^{3}\mathbb{I}_{ijx\gamma} = W_{\zeta\gamma}^{'} \mathbf{1}_{j}^{'} {}^{3}\mathbb{P}_{ix\zeta} + W_{\zeta\gamma}^{''} \mathbf{1}_{i}^{''} {}^{3}\mathbb{P}_{jx\zeta}
    \end{aligned}
    $$


    `non-weighted`:
    $$
    \begin{aligned}
    {}^{3}\mathbb{I}_{ijx\zeta} = \mathbf{1}_{j} {}^{3}\mathbb{P}_{ix\zeta}
    \end{aligned}
    $$

    """

    def __init__(self, n_nodes, weighted: bool, **kwargs):
        """
        Args:
            weighted (bool): style of the layer, should be a bool
        """
        super(PIXLayer, self).__init__()
        self.n_nodes = n_nodes
        self.weighted = weighted

    def build(self, shapes):
        if self.weighted:
            self.wi = tf.keras.layers.Dense(
                shapes[1][-1], activation=None, use_bias=False
            )
            self.wj = tf.keras.layers.Dense(
                shapes[1][-1], activation=None, use_bias=False
            )

        self.ff_layer = FFLayer(self.n_nodes, activation=None, use_bias=False)

    def call(self, tensors):
        """
        PILayer take a list of three tensors as input:

        - ind_2: [sparse indices](layers.md#sparse-indices) of pairs with shape `(n_pairs, 2)`
        - prop: equalvariant tensor with shape `(n_atoms, x, n_prop)`

        Args:
            tensors (list of tensors): list of `[ind_2, prop]` tensors

        Returns:
            inter (tensor): interaction tensor with shape `(n_pairs, x, n_nodes[-1])`
        """
        ind_2, px = tensors
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        px_i = tf.gather(px, ind_i)
        px_j = tf.gather(px, ind_j)

        if self.weighted:
            return self.ff_layer(self.wi(px_i) + self.wj(px_j))
        else:
            return self.ff_layer(px_j)


class GCBlock(tf.keras.layers.Layer):
    def __init__(self, weighted: bool, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super(GCBlock, self).__init__()
        iiargs = kwargs.copy()
        iiargs.update(use_bias=False)
        ii1_nodes = ii_nodes.copy()
        pp1_nodes = pp_nodes.copy()
        ii1_nodes[-1] *= 5
        pp1_nodes[-1] = ii_nodes[-1] * 3
        self.pp1_layer = FFLayer(pp1_nodes, **kwargs)
        self.pi1_layer = PILayer(pi_nodes, **kwargs)
        self.ii1_layer = FFLayer(ii1_nodes, **iiargs)
        self.ip1_layer = IPLayer()

        ppx_nodes = [pp_nodes[-1]]

        self.pp3_layer = FFLayer(ppx_nodes, activation=None, use_bias=False)
        self.pi3_layer = PIXLayer(ppx_nodes, weighted=weighted, **kwargs)
        self.ip3_layer = IPLayer()

        self.pp5_layer = FFLayer(ppx_nodes, activation=None, use_bias=False)
        self.pi5_layer = PIXLayer(ppx_nodes, weighted=weighted, **kwargs)
        self.ip5_layer = IPLayer()

        self.dot_layer1 = DotLayer(weighted=weighted)
        self.dot_layer2 = DotLayer(weighted=weighted)

        self.scale0_layer = ScaleLayer()
        self.scale1_layer = ScaleLayer()
        self.scale2_layer = ScaleLayer()
        self.scale3_layer = ScaleLayer()
        self.scale4_layer = ScaleLayer()

    def build(self, tensor):
        pass

    def call(self, tensors):
        ind_2, p1, p3, p5, diff, diff_p5, basis = tensors
        
        i1 = self.pi1_layer([ind_2, p1, basis])
        i1 = self.ii1_layer(i1)
        i1_1, i1_2, i1_3, i1_4, i1_5 = tf.split(i1, 5, axis=-1)
        p1 = self.ip1_layer([ind_2, p1, i1_1])
        scaled_diff3 = self.scale0_layer([diff[:, :, None], i1_2])
        scaled_diff5 = self.scale1_layer([diff_p5[:, :, None], i1_3])

        i3 = self.pi3_layer([ind_2, p3])

        i3 = self.scale1_layer([i3, i1_4])
        i3 = i3 + scaled_diff3
        p3 = self.ip3_layer([ind_2, p3, i3])

        
        i5 = self.pi5_layer([ind_2, p5])
        i5 = self.scale2_layer([i5, i1_5])

        i5 = i5 + scaled_diff5
        p5 = self.ip5_layer([ind_2, p5, i5])

        p1t1 = tf.concat([self.dot_layer1(tf.reshape(p5, (-1, 5, p5.shape[-1]))), self.dot_layer2(p3), p1], axis=1)
        p1t1 = self.pp1_layer(p1t1)
        p1t1, p1t1_2, p1t1_3 = tf.split(p1t1, 3, axis=-1)

        p3t1 = self.scale3_layer([p3, p1t1_2])
        p3t1 = self.pp3_layer(p3t1)

        p5t1 = self.scale4_layer([p5, p1t1_3])
        p5t1 = self.pp5_layer(p5t1)

        return p1t1, p3t1, p5t1


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, atom_types, rc):
        super(PreprocessLayer, self).__init__()
        self.embed = AtomicOnehot(atom_types)
        self.nl_layer = CellListNL(rc)

    def call(self, tensors):
        tensors = tensors.copy()
        for k in ["elems", "dist"]:
            if k in tensors.keys():
                tensors[k] = tf.reshape(tensors[k], tf.shape(tensors[k])[:1])
        if "ind_2" not in tensors:
            tensors.update(self.nl_layer(tensors))
            tensors["p1"] = tf.cast(  # difference with pinet: prop->p1
                self.embed(tensors["elems"]), tensors["coord"].dtype
            )
        tensors["norm_diff"] = tensors["diff"] / tf.linalg.norm(tensors["diff"])
        diff = tensors["norm_diff"]
        x = diff[:, 0]
        y = diff[:, 1]
        z = diff[:, 2]
        x2 = x**2
        y2 = y**2
        z2 = z**2
        tensors["diff_p5"] = tf.stack([
            2/3 * x2 - 1/3 * y2 - 1/3 * z2,
            2/3 * y2 - 1/3 * x2 - 1/3 * z2,
            x*y,
            x*z,
            y*z
        ], axis=1)
        return tensors


class PiNet2P5Dot_i(tf.keras.Model):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        atom_types=[1, 6, 7, 8],
        rc=4.0,
        cutoff_type="f1",
        basis_type="polynomial",
        n_basis=4,
        gamma=3.0,
        center=None,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        out_units=1,
        out_prop=1,
        out_inter=0,
        out_pool=False,
        act="tanh",
        depth=4,
        weighted=True,
    ):
        """
        Args:
            atom_types (list): elements for the one-hot embedding
            pp_nodes (list): number of nodes for PPLayer
            pi_nodes (list): number of nodes for PILayer
            ii_nodes (list): number of nodes for IILayer
            out_nodes (list): number of nodes for OutLayer
            out_pool (str): pool atomic outputs, see ANNOutput
            depth (int): number of interaction blocks
            rc (float): cutoff radius
            basis_type (string): basis function, can be "polynomial" or "gaussian"
            n_basis (int): number of basis functions to use
            gamma (float or array): width of gaussian function for gaussian basis
            center (float or array): center of gaussian function for gaussian basis
            cutoff_type (string): cutoff function to use with the basis.
            act (string): activation function to use
            weighted (bool): whether to use weighted style
        """
        super(PiNet2P5Dot_i, self).__init__()

        self.depth = depth
        self.preprocess = PreprocessLayer(atom_types, rc)
        self.cutoff = CutoffFunc(rc, cutoff_type)

        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        self.res_update1 = [ResUpdate() for i in range(depth)]
        self.res_update3 = [ResUpdate() for i in range(depth)]
        self.res_update5 = [ResUpdate() for i in range(depth)]
        self.gc_blocks = [
            GCBlock(weighted, pp_nodes, pi_nodes, ii_nodes, activation=act)
            for _ in range(depth)
        ]

        self.pout_layers = [OutLayer(out_nodes, out_units) for i in range(depth)]

        if out_inter>0:
            self.iout_layers  = [PILayer(out_nodes+[out_inter]) for i in range(depth)]
            self.iout3_layers = [PIXLayer(out_nodes+[out_inter], weighted=weighted) for i in range(depth)]
            self.iout5_layers = [PIXLayer(out_nodes+[out_inter], weighted=weighted) for i in range(depth)]
            self.dot_layer1   = DotLayer(weighted=weighted)
            self.dot_layer2   = DotLayer(weighted=weighted)
        else:
            self.iout_layers = None

        self.out_pool = out_pool
        self.ann_output = ANNOutput(out_pool)

    def build(self, tensors):
        pass

    def call(self, tensors):
        """PiNet takes batches atomic data as input, the following keys are
        required in the input dictionary of tensors:

        - `ind_1`: [sparse indices](layers.md#sparse-indices) for the batched data, with shape `(n_atoms, 1)`;
        - `elems`: element (atomic numbers) for each atom, with shape `(n_atoms)`;
        - `coord`: coordintaes for each atom, with shape `(n_atoms, 3)`.

        Optionally, the input dataset can be processed with
        `PiNet.preprocess(tensors)`, which adds the following tensors to the
        dictionary:

        - `ind_2`: [sparse indices](layers.md#sparse-indices) for neighbour list, with shape `(n_pairs, 2)`;
        - `dist`: distances from the neighbour list, with shape `(n_pairs)`;
        - `diff`: distance vectors from the neighbour list, with shape `(n_pairs, 3)`;
        - `prop`: initial properties `(n_pairs, n_elems)`;

        Args:
            tensors (dict of tensors): input tensors

        Returns:
            output (tensor): output tensor with shape `[n_atoms, out_nodes]`
        """
        tensors = self.preprocess(tensors)
        tensors["p3"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 3, 1])
        tensors["p5"] = tf.zeros([tf.shape(tensors["ind_1"])[0], 5, 1])
        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        
        if self.iout_layers is not None and self.out_pool==True:
            raise Exception("Currently this is not implemented in PiNN.")

        if self.iout_layers is not None:
            pout = 0.0
            iout = 0.0
            iout3 = 0.0
            iout5 = 0.0

            for i in range(self.depth):
                p1, p3, p5 = self.gc_blocks[i](
                            [tensors["ind_2"], tensors["p1"], tensors["p3"], tensors["p5"], tensors["norm_diff"], tensors["diff_p5"], basis]
            )
                pout = self.pout_layers[i]([tensors["ind_1"], p1, p3, pout])
                
                i1 = self.iout_layers[i]([tensors["ind_2"], p1, basis])
                iout += i1

                i3 = self.iout3_layers[i]([tensors["ind_2"], p3])
                iout3 += self.dot_layer2(i3) + i1

                i5 = self.iout5_layers[i]([tensors["ind_2"], p5])
                iout5 += tf.concat([self.dot_layer1(tf.reshape(i5, (-1, 5, i5.shape[-1]))), self.dot_layer2(i3), i1], axis=1)

                tensors["p1"] = self.res_update1[i]([tensors["p1"], p1])
                tensors["p3"] = self.res_update3[i]([tensors["p3"], p3])
                tensors["p5"] = self.res_update5[i]([tensors["p5"], p5])

            pout = self.ann_output([tensors["ind_1"], pout])

            return pout, iout5
        else:
            pout = 0.0
            for i in range(self.depth):
                p1, p3, p5 = self.gc_blocks[i](
                            [tensors["ind_2"], tensors["p1"], tensors["p3"], tensors["p5"], tensors["norm_diff"], tensors["diff_p5"], basis]
            )
                pout = self.pout_layers[i]([tensors["ind_1"], p1, p3, pout])
                tensors["p1"] = self.res_update1[i]([tensors["p1"], p1])
                tensors["p3"] = self.res_update3[i]([tensors["p3"], p3])
                tensors["p5"] = self.res_update5[i]([tensors["p5"], p5])

            pout = self.ann_output([tensors["ind_1"], pout])
            return pout