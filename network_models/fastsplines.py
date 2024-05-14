# fastsplines
# fix highest power = 3 and gridpoints to 10

# all things considered, this fixed activation function design is good enough.
# we ingest a set of hyperparameters and deduce the corresponding activation function from it

# faster without class

def activate(hyperparameters):
    # hyperparameters[] = [w, c3, c2, c1, c0] - enough for 1 edge
