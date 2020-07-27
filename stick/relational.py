from .stick import *


class Minimum(STICK):
    def __init__(self):
        super(Minimum, self).__init__()

        input1, input2 = self.create_neuron(['input1', 'input2'])
        smaller1, smaller2 = self.create_neuron(['smaller1', 'smaller2'])
        output = self.create_neuron('output')

        # Set the inputs
        self.set_input('input1', input1)
        self.set_input('input2', input2)

        # Form the internal connections
        self.connect(input1, smaller1, weight=0.5 * w_e)
        self.connect(input1, output, weight=0.5 * w_e, delay=2 * T_syn + T_neu)
        self.connect(input2, smaller2, weight=0.5 * w_e)
        self.connect(input2, output, weight=0.5 * w_e, delay=2 * T_syn + T_neu)

        self.connect(smaller1, output, weight=0.5 * w_e)
        self.connect(smaller1, input2, weight=w_i)
        self.connect(smaller1, smaller2, weight=0.5 * w_i)

        self.connect(smaller2, output, weight=0.5 * w_e)
        self.connect(smaller2, input1, weight=w_i)
        self.connect(smaller2, smaller1, weight=0.5 * w_i)

        # Set the outputs
        self.set_output('output', output)
        self.set_output('smaller1', smaller1)
        self.set_output('smaller2', smaller2)


class Maximum(STICK):
    def __init__(self):
        super(Maximum, self).__init__()

        input1, input2 = self.create_neuron(['input1', 'input2'])
        larger1, larger2 = self.create_neuron(['larger1', 'larger2'])
        output = self.create_neuron('output')

        # Set the inputs
        self.set_input('input1', input1)
        self.set_input('input2', input2)

        # Form the internal connections
        self.connect(input1, larger2, weight=0.5 * w_e)
        self.connect(input1, output, weight=0.5 * w_e)
        self.connect(input2, larger1, weight=0.5 * w_e)
        self.connect(input2, output, weight=0.5 * w_e)

        self.connect(larger1, larger2, weight=0.5 * w_i)
        self.connect(larger2, larger1, weight=0.5 * w_i)

        # Set the outputs
        self.set_output('output', output)
        self.set_output('larger1', larger1)
        self.set_output('larger2', larger2)
