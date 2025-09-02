class BaseCallback:
    """Base class for training callbacks.

    Methods mirror transformers.TrainerCallback but are framework-agnostic.
    Subclasses can override any subset without needing to depend on transformers.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass


class ConsoleCallback(BaseCallback):
    """Simple console logger callback."""

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            print("--- Training started ---")
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            if logs:
                step = getattr(state, 'global_step', None)
                msg = f"step={step} | logs={logs}"
                print(msg)
        except Exception:
            pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        try:
            print(f"Epoch {int(state.epoch) if getattr(state, 'epoch', None) is not None else 0} begin")
        except Exception:
            pass

    def on_step_begin(self, args, state, control, **kwargs):
        try:
            print(f"Step {getattr(state, 'global_step', 0) + 1} begin")
        except Exception:
            pass

    def on_step_end(self, args, state, control, **kwargs):
        try:
            print(f"Step {getattr(state, 'global_step', 0)} end")
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        try:
            print("--- Training finished ---")
        except Exception:
            pass


