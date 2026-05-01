function ghz_state = generate_ghz(n)
    % Generate the n-qubit GHZ state |GHZ> = (|00...0> + |11...1>) / sqrt(2)

    % Define the basis states |0> and |1>
    zero_state = [1; 0];
    one_state = [0; 1];

    % Generate |00...0> and |11...1> states using tensor product
    state_0 = zero_state;
    state_1 = one_state;

    for i = 2:n
        state_0 = kron(state_0, zero_state); % |00...0>
        state_1 = kron(state_1, one_state); % |11...1>
    end

    % Construct the GHZ state
    ghz_state = (state_0 + state_1) / sqrt(2);

    % Display the result
    disp('GHZ State Vector:');
    disp(ghz_state);
end
