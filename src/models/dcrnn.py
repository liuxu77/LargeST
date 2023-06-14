import torch
import torch.nn as nn
import numpy as np
from src.base.model import BaseModel
from src.utils.graph_algo import normalize_adj_mx

class DCRNN(BaseModel):
    '''
    Reference code: https://github.com/chnsh/DCRNN_PyTorch
    '''
    def __init__(self, device, adj_mx, n_filters, max_diffusion_step, filter_type, \
                 num_rnn_layers, cl_decay_steps, use_curriculum_learning=True, **args):
        super(DCRNN, self).__init__(**args)
        self.device = device
        self.supports = self._calculate_supports(adj_mx, filter_type)

        self.encoder = DCRNNEncoder(device=device,
                                    node_num=self.node_num,
                                    input_dim=self.input_dim,
                                    hid_dim=n_filters,
                                    max_diffusion_step=max_diffusion_step,
                                    filter_type=filter_type,
                                    num_rnn_layers=num_rnn_layers)

        self.decoder = DCGRUDecoder(device=device,
                                    node_num=self.node_num,
                                    input_dim=self.output_dim,
                                    hid_dim=n_filters,
                                    output_dim=self.output_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    filter_type=filter_type,
                                    num_rnn_layers=num_rnn_layers)

        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps


    def _calculate_supports(self, adj_mx, filter_type):
        supports = normalize_adj_mx(adj_mx, filter_type, 'coo')

        results = []
        for support in supports:
            results.append(self._build_sparse_matrix(support).to(self.device))
        return results


    def _build_sparse_matrix(self, L):
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))


    def forward(self, source, target, iter=None):  # (b, t, n, f)
        b, t, n, _ = source.shape
        go_symbol = torch.zeros(
            1, b, self.node_num, self.output_dim).to(self.device)

        source = torch.transpose(source, dim0=0, dim1=1)

        target = torch.transpose(
            target[..., :self.output_dim], dim0=0, dim1=1)
        target = torch.cat([go_symbol, target], dim=0)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(b).to(self.device)

        # last hidden state of the encoder is the context
        context, _ = self.encoder(source, self.supports, init_hidden_state)

        if self.training and self.use_curriculum_learning:
            teacher_forcing_ratio = self._compute_sampling_threshold(iter)
        else:
            teacher_forcing_ratio = 0

        outputs = self.decoder(
            target, self.supports, context, teacher_forcing_ratio=teacher_forcing_ratio)
        o = outputs[1:, :, :].permute(1, 0, 2).reshape(b, t, n, self.output_dim)
        return o


class DCRNNEncoder(nn.Module):
    def __init__(self, device, node_num, input_dim, hid_dim, \
                 max_diffusion_step, filter_type, num_rnn_layers):
        super(DCRNNEncoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers

        encoding_cells = list()
        encoding_cells.append(DCGRUCell(device=device,
                                        node_num=node_num,
                                        input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        filter_type=filter_type))

        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(device=device,
                                            node_num=node_num,
                                            input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            filter_type=filter_type))

        self.encoding_cells = nn.ModuleList(encoding_cells)


    def forward(self, inputs, supports, initial_hidden_state):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(
            inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    current_inputs[t, ...], supports, hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)

            current_inputs = torch.stack(output_inner, dim=0)
        return output_hidden, current_inputs


    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, device, node_num, input_dim, hid_dim, output_dim, \
                 max_diffusion_step, filter_type, num_rnn_layers):
        super(DCGRUDecoder, self).__init__()
        self.device = device
        self.node_num = node_num
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers

        decoding_cells = list()
        decoding_cells.append(DCGRUCell(device=device,
                                        node_num=node_num,
                                        input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        filter_type=filter_type))

        for _ in range(1, num_rnn_layers - 1):
            decoding_cells.append(DCGRUCell(device=device,
                                            node_num=node_num,
                                            input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            filter_type=filter_type))
        
        cell_with_projection = DCGRUCell(device=device,
                                         node_num=node_num,
                                         input_dim=hid_dim,
                                         num_units=hid_dim,
                                         max_diffusion_step=max_diffusion_step,
                                         filter_type=filter_type,
                                         num_proj=output_dim)

        decoding_cells.append(cell_with_projection)
        self.decoding_cells = nn.ModuleList(decoding_cells)


    def forward(self, inputs, supports, initial_hidden_state, teacher_forcing_ratio=0.5):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(
            inputs, (seq_length, batch_size, -1))

        outputs = torch.zeros(
            seq_length, batch_size, self.node_num*self.output_dim).to(self.device)

        current_input = inputs[0]
        for t in range(1, seq_length):
            next_input_hidden_state = []

            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    current_input, supports, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)

            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            current_input = (inputs[t] if teacher_force else output)
        return outputs


class DCGRUCell(nn.Module):
    def __init__(self, device, node_num, input_dim, num_units, max_diffusion_step, \
                 filter_type, num_proj=None, activation=torch.tanh, use_gc_for_ru=True):
        super(DCGRUCell, self).__init__()
        self.device = device
        self.node_num = node_num
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._activation = activation
        self._use_gc_for_ru = use_gc_for_ru

        if filter_type == 'doubletransition':
            supports_len = 2
        else:
            supports_len = 1

        self.dconv_gate = DiffusionGraphConv(node_num=node_num,
                                             supports_len=supports_len,
                                             input_dim=input_dim,
                                             hid_dim=num_units,
                                             output_dim=num_units*2,
                                             max_diffusion_step=max_diffusion_step)

        self.dconv_candidate = DiffusionGraphConv(node_num=node_num,
                                                  supports_len=supports_len,
                                                  input_dim=input_dim,
                                                  hid_dim=num_units,
                                                  output_dim=num_units,
                                                  max_diffusion_step=max_diffusion_step)

        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)


    @property
    def output_size(self):
        output_size = self.node_num * self._num_units
        if self._num_proj is not None:
            output_size = self.node_num * self._num_proj
        return output_size


    def forward(self, inputs, supports, state):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(inputs, supports, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self.node_num, output_size))

        r, u = torch.split(
            value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, self.node_num * self._num_units))
        u = torch.reshape(u, (-1, self.node_num * self._num_units))
        c = self.dconv_candidate(inputs, supports, r * state, self._num_units)

        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * state + (1 - u) * c

        if self._num_proj is not None:
            batch_size = inputs.shape[0]
            output = torch.reshape(new_state, shape=(-1, self._num_units))
            output = torch.reshape(self.project(output), shape=(
                batch_size, self.output_size))
        return output, new_state


    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)


    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass


    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.node_num * self._num_units).to(self.device)
    

class DiffusionGraphConv(nn.Module):
    def __init__(self, node_num, supports_len, input_dim, hid_dim, \
                 output_dim, max_diffusion_step, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.node_num = node_num
        self.num_matrices = supports_len * max_diffusion_step + 1
        input_size = input_dim + hid_dim
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(
            size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))

        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)


    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)


    def forward(self, inputs, supports, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.node_num, -1))
        state = torch.reshape(state, (batch_size, self.node_num, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)
        x0 = torch.reshape(
            x0, shape=[self.node_num, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, self.node_num, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)
        x = torch.reshape(
            x, shape=[batch_size * self.node_num, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self.node_num * output_size])