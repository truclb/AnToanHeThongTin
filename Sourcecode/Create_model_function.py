def create_ann_model(input_shape=(94,), hidden_units=None):
    """
    Create a feedforward neural network model for binary classification with
    the specified input shape.

    Parameters:
    - input_shape: Tuple specifying the shape of the input data.
    - hidden_units: List of integers specifying the number of units in each hidden layer.
                    Default is [64, 32].

    Returns:
    - A compiled Keras model.
    """

    if hidden_units is None:
        hidden_units = [64, 32]

    model = Sequential()

    # Input layer
    model.add(Dense(hidden_units[0], activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))

    # Hidden layers
    for units in hidden_units[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model