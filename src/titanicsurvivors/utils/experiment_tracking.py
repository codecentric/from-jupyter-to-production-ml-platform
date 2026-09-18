from zenml.client import Client


def get_experiment_tracker_name() -> str | None:
    experiment_tracker = Client().active_stack.experiment_tracker
    if experiment_tracker is not None:
        return experiment_tracker.name
    return None
