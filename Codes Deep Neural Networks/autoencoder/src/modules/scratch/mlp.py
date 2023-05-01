import numpy as np


class MultilayerPerceptron:

    def __init__(self, architecture, activations, learning_rate=0.5, add_bias=True):
        """
            Initialize Multilayer Perceptron

            Parameters
            ----------
            architecture : list
                network architecture, i.e., [l0, l1, ..., lL]
                where, li is the number of neurons at layer i = 0,...,L
            activations : function
                activation functions
            learning_rate : float
                learning rate
        """
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.activations = activations
        self.learning_rate = learning_rate
        self.add_bias = add_bias
        self.__boot()

    def __boot(self, low=-1.0, high=1.0):
        """
            Select starting weights randomly

            Parameters
            ----------
            low : float
                lower bound for initial weight assignment
            high : float
                upper bound for initial weight assignment

            Notes:
            - weights[k][i, j], is the synaptic weight connecting the
                i-th neuron of layer k to the j-th neuron of layer k + 1
        """
        self.weights = []
        extra_weight = 1 if self.add_bias else 0
        for layer in range(self.num_layers - 1):
            weight_matrix = np.random.uniform(low, high, size=(
                self.architecture[layer] + extra_weight, self.architecture[layer + 1]))
            self.weights.append(weight_matrix)

    def __forward(self, instance):
        """
            Perform a forward pass

            Parameters
            ----------
            instance : list
                sample to forward pass
        """
        fields = []
        for layer in range(self.num_layers - 1):
            # Append bias
            if self.add_bias:
                instance = np.append(instance, 1.0)
            # Compute local fields
            local_fields = np.matmul(instance, self.weights[layer])
            fields.append(local_fields)
            # Apply activation function
            eval_activation, _ = self.activations[layer]()
            instance = eval_activation(local_fields)
        return instance, fields

    def __compute_local_gradients(self, error_signal, fields):
        """
            Compute local gradients

            Parameters
            ----------
            error_signal : list
                error signal measured at the output layer
            fields : list
                local fields computed during forward pass
        """
        layers_gradients = []
        layers_sums = []
        for layer in reversed(range(1, self.num_layers)):
            _, eval_activation_diff = self.activations[layer - 1]()
            local_gradients = []
            for j in range(self.architecture[layer]):
                if layer == self.num_layers - 1:
                    # Compute local gradient for output layer
                    delta = error_signal[j] * \
                        eval_activation_diff(fields[layer - 1][j])
                else:
                    # Compute local gradient for hidden layer
                    next_deltas = layers_gradients[-1]
                    omega = 0.0
                    for k in range(self.architecture[layer + 1]):
                        omega += next_deltas[k] * self.weights[layer][j, k]
                    delta = eval_activation_diff(fields[layer - 1][j]) * omega
                local_gradients.append(delta)
            # Update layers' gradients
            layers_gradients.append(local_gradients)
            # Compute local gradients' sum
            local_gradients_sum = np.sum(np.abs(local_gradients))
            # Update layers' sums
            layers_sums.append(local_gradients_sum)
        layers_gradients.reverse()
        layers_sums.reverse()
        return layers_gradients, layers_sums

    def __update_weights(self, local_gradients, fields, instance):
        """
            Update weights according to delta rule

            Parameters
            ----------
            local_gradients : list
                calculated local gradients
            fields : list
                local fields computed during forward pass
            instance : list
                current sample
        """
        for layer in range(self.num_layers - 1):
            eval_activation, _ = self.activations[layer]()
            # Compute output signal
            if layer == 0:
                output_signal = instance
            else:
                output_signal = eval_activation(fields[layer - 1])
            if self.add_bias:
                output_signal = np.append(output_signal, 1.0)
            # Update weights
            extra_weight = 1 if self.add_bias else 0
            for i in range(self.architecture[layer] + extra_weight):
                for j in range(self.architecture[layer + 1]):
                    # Apply delta rule
                    self.weights[layer][i, j] = self.weights[layer][i, j] + \
                        self.learning_rate * \
                        local_gradients[layer][j] * output_signal[i]

    def fit(self, data, labels, epochs=50, tol=1e-2, stop_tol=True):
        """
            Fit Multilayer Perceptron

            Parameters
            ----------
            data : list
                training samples
            labels : list
                training labels
            epochs : int
                epochs
            tol : float
                tolerance
        """
        # Number of samples
        num_samples = len(data)
        # Network's average errors
        network_avg_errors = []
        # Gradients' sums
        gradients_sums = []
        # Backpropagation
        for epoch in range(epochs):
            # Shuffle training samples
            perm = np.random.permutation(num_samples)
            data = data[perm, :]
            if labels.ndim > 1:
                labels = labels[perm, :]
            else:
                labels = labels[perm]
            # Total error energy
            total_error_energy = np.array([0.0] * num_samples)
            # Epoch's local gradients sum
            epoch_local_gradients_sum = np.array([0.0] * (self.num_layers - 1))
            # Sequential model
            for n in range(num_samples):
                instance = data[n, :]
                # Forward pass
                output, fields = self.__forward(instance)
                # Compute error signal
                if labels.ndim > 1:
                    error_signal = labels[n, :] - output
                else:
                    error_signal = labels[n] - output
                # Compute total error energy
                total_error_energy[n] = 0.5 * np.sum(error_signal ** 2)
                # Compute local gradients
                local_gradients, layers_sums = self.__compute_local_gradients(
                    error_signal, fields)
                # Update epoch's local gradients sum
                epoch_local_gradients_sum += layers_sums
                # Update weights
                self.__update_weights(local_gradients, fields, instance)
            # Compute average energy error
            avg_energy_error = np.mean(total_error_energy)
            network_avg_errors.append(avg_energy_error)
            # Update gradients' sums
            gradients_sums.append(epoch_local_gradients_sum / num_samples)
            # Stopping condition
            if stop_tol and network_avg_errors[epoch] <= tol:
                break
        return np.array(network_avg_errors), np.array(gradients_sums)

    def predict(self, data):
        """
            Predict labels

            Parameters
            ----------
            data : list
                test samples
        """
        outputs = []
        for instance in data:
            output, _ = self.__forward(instance)
            outputs.append(output)
        return np.array(outputs)
