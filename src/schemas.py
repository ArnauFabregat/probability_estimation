from pydantic import BaseModel, Field, PositiveInt, PositiveFloat


class OptimizerParams(BaseModel):
    """
    Hyperparameters used to configure the optimizer and early‑stopping
    behavior during training.

    Parameters
    ----------
    lr : PositiveFloat, default=1e-3
        Learning rate for the optimizer.

    weight_decay : float, default=1e-4
        L2 regularization coefficient applied to the optimizer.
        Must be non‑negative.

    batch_size : PositiveInt, default=1024
        Minibatch size used during training.

    patience : PositiveInt, default=10
        Number of consecutive epochs without improvement in validation loss
        before early stopping is triggered.

    min_delta : float, default=1e-4
        Minimum required reduction in validation loss to count as an
        improvement. Smaller changes are ignored for early stopping.
    """
    lr: PositiveFloat = 1e-3
    weight_decay: float = Field(default=1e-4, ge=0)
    batch_size: PositiveInt = 1024
    patience: PositiveInt = 10
    min_delta: float = 1e-4
